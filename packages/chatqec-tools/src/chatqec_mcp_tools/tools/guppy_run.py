"""guppy_run tool -- run a Guppy quantum program on the Selene emulator and return shot counts.

SECURITY: this executes a program supplied through chat. There is no OS sandbox (unprivileged user
namespaces are blocked on many machines), so three layers stand in for one:

  1. OFF BY DEFAULT. It runs only when CHATQEC_ALLOW_GUPPY_EXEC=1 is set in the environment of the
     MCP server, and the ``[guppy]`` extra is installed.
  2. A SOURCE GATE (_guppy_gate.py) allows only guppylang imports and ``@guppy`` functions, so no
     other top-level Python can run, and bodies may not use ``py(...)``/``comptime``/dunders.
  3. An ISOLATED CHILD (_guppy_exec.py): scrubbed environment, private temp directory, CPU, file
     and memory limits, and a process-group kill on timeout.

None of this blocks network access or reads of the user's files from a bug in guppylang itself.
Enable it only for input you would be willing to run.

Probed with guppylang 0.21.16 / selene-sim 0.2.18: ``main.emulator(n_qubits=N).with_shots(S)
.with_seed(k).run()`` returns shots whose ``entries`` are ``(register, value)`` pairs. A cold run
takes about 12 s (Selene builds with a bundled Zig toolchain); Zig needs about 8 GB of address space.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._guppy_exec import ExecResult, run_isolated
from chatqec_mcp_tools.tools._guppy_gate import validate_guppy_source

_ENV_FLAG = "CHATQEC_ALLOW_GUPPY_EXEC"
_WALL_SECONDS = 55  # inside the 60 s MCP call budget
_CPU_SECONDS = 100
_MEMORY_MB = 8192
_MAX_DISTINCT_OUTCOMES = 1_000
_MARKER = "@@GUPPY_RESULT@@"
_INSTALL_HINT = "uv tool install 'chatqec-mcp-tools[guppy]'"
_EPILOGUE = """

import json as _json

_emulator = main.emulator(n_qubits={n_qubits}).with_shots({shots})
{seed_line}
_shots = [dict(_s.entries) for _s in _emulator.run().results]
_registers = []
for _entries in _shots:
    for _tag in _entries:
        if _tag not in _registers:
            _registers.append(_tag)
_counts = {{}}
for _entries in _shots:
    _key = ",".join(str(_entries.get(_tag)) for _tag in _registers)
    _counts[_key] = _counts.get(_key, 0) + 1
print("{marker}" + _json.dumps({{"registers": _registers, "counts": _counts}}, default=str))
"""


class GuppyRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(
        ...,
        max_length=20_000,
        description=(
            "A Guppy program: guppylang imports and @guppy functions only, defining main(). "
            "Report outcomes with result('name', value)"
        ),
    )
    shots: int = Field(100, ge=1, le=10_000, description="Number of shots")
    n_qubits: int = Field(4, ge=1, le=24, description="Qubits available to the emulator")
    seed: int | None = Field(default=None, description="Emulator seed for reproducibility")


class GuppyRunOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    registers: list[str] = []
    counts: dict[str, int] = Field(
        default_factory=dict, description="Outcome (register values joined by commas) to shots"
    )
    shots: int = 0
    notes: str = ""
    stderr_tail: str = ""
    error_message: str = ""


def _fail(message: str, stderr_tail: str = "") -> GuppyRunOutput:
    return GuppyRunOutput(ok=False, error_message=message, stderr_tail=stderr_tail)


def _precheck(inp: GuppyRunInput) -> GuppyRunOutput | None:
    """A failure result when the run must not start, else None."""
    if os.environ.get(_ENV_FLAG) != "1":
        return _fail(
            "guppy_run is disabled: it executes the submitted program and there is no OS "
            f"sandbox. Set {_ENV_FLAG}=1 in the MCP server's environment to enable it."
        )
    if (
        importlib.util.find_spec("guppylang") is None
        or importlib.util.find_spec("selene_sim") is None
    ):
        return _fail(f"guppylang / selene-sim are not installed: {_INSTALL_HINT}")
    try:
        validate_guppy_source(inp.source)
    except ValueError as exc:
        return _fail(f"program rejected: {exc}")
    return None


def _read_result(run: ExecResult, shots: int) -> GuppyRunOutput:
    tail = run.stderr[-800:]
    if run.timed_out:
        return _fail(f"the program exceeded the {_WALL_SECONDS} s time limit and was stopped", tail)
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith(_MARKER)]
    if run.exit_code != 0 or not lines:
        return _fail(f"the program failed (exit code {run.exit_code}); see stderr_tail", tail)
    try:
        data = json.loads(lines[-1][len(_MARKER) :])
    except json.JSONDecodeError:
        return _fail("the emulator produced an unreadable result", tail)
    counts = dict(Counter(data["counts"]).most_common(_MAX_DISTINCT_OUTCOMES))
    notes = ""
    if len(data["counts"]) > _MAX_DISTINCT_OUTCOMES:
        notes = f"only the {_MAX_DISTINCT_OUTCOMES} most frequent outcomes are listed"
    return GuppyRunOutput(
        ok=True, registers=data["registers"], counts=counts, shots=shots, notes=notes
    )


def guppy_run(inp: GuppyRunInput) -> GuppyRunOutput:
    refused = _precheck(inp)
    if refused is not None:
        return refused
    seed_line = f"_emulator = _emulator.with_seed({inp.seed})" if inp.seed is not None else ""
    program = inp.source + _EPILOGUE.format(
        n_qubits=inp.n_qubits, shots=inp.shots, seed_line=seed_line, marker=_MARKER
    )
    run = run_isolated(
        program, wall_seconds=_WALL_SECONDS, cpu_seconds=_CPU_SECONDS, memory_mb=_MEMORY_MB
    )
    return _read_result(run, inp.shots)
