# `code_params` Tool (Stim-backed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a `code_params` MCP tool that reports a named QEC code's parameters ([[n,k]], stabilizer count) plus a **Stim-computed** code distance — replacing the deferred `qiskit_qec_run` with a clean, dependency-free implementation.

**Architecture:** Pure `fn(Input)->Output` in the `chatqec-mcp-tools` sibling, same pattern as the other tools. `n`, `k`, `num_stabilizers` come from each family's standard formula; the distance is computed by `stim.Circuit.generated(...).shortest_graphlike_error()`. Then wire it into the main app's prompt + Streamlit rendering.

**Tech Stack:** Python 3.11, pydantic 2, stim (already a dep), pytest.

## Global Constraints
- Two repos: sibling `~/chatqec-mcp-tools` (Task A), main `~/chatQEC` (Task B). Sibling commands use its `.venv`/`uv`.
- Tool contract: pydantic `Input`/`Output` (`ConfigDict(extra="forbid")`); always `ok: bool` + `error_message: str = ""`; NO exception escapes the pure fn. Lazy-import `stim` with `# noqa: PLC0415`.
- Distance guards: `distance` in 3..15; `surface_code_rotated` and `color_code` require **odd** distance (`repetition_code` allows any).
- Family → stim generator: `surface_code_rotated → "surface_code:rotated_memory_z"`, `repetition_code → "repetition_code:memory"`, `color_code → "color_code:memory_xyz"`.
- `n` formula per family: surface `d*d`; repetition `d`; color `(3*d*d+1)//4`. `k = 1` for all three. `num_stabilizers = n - k`.
- Color-code note: `shortest_graphlike_error` gives the **graphlike** distance (a lower bound; hyperedge errors undercounted) — surface it in a `notes` field.
- Ruff `E,F,I,B,UP,SIM,PL,RUF` line-length 100; mypy strict on `src`. Tests may ignore `PLR2004,S101,I001,PLC0415`. Conventional Commits on `main`, no `Co-Authored-By`.
- `server.py` registration follows the EXACT existing pattern (import, `_<name>_handler`, append `ToolSpec`).

---

### Task 1: `code_params` tool + registration + README (sibling)

**Files:**
- Create: `~/chatqec-mcp-tools/src/chatqec_mcp_tools/tools/code_params.py`
- Modify: `~/chatqec-mcp-tools/src/chatqec_mcp_tools/server.py`
- Modify: `~/chatqec-mcp-tools/README.md`
- Test: `~/chatqec-mcp-tools/tests/test_code_params.py`

**Interfaces produced:** `CodeParamsInput(code_family: Literal["surface_code_rotated","repetition_code","color_code"], distance: int)`, `CodeParamsOutput(ok, code_family, distance, n, k, num_stabilizers, stim_code_distance, num_detectors, num_observables, notes, error_message)`, `code_params(inp) -> CodeParamsOutput`.

- [ ] **Step 1: Write the failing test** — `tests/test_code_params.py`:
```python
"""code_params tool tests (values verified against real Stim output)."""

from chatqec_mcp_tools.tools.code_params import (
    CodeParamsInput,
    CodeParamsOutput,
    code_params,
)


def test_surface_code_params():
    out = code_params(CodeParamsInput(code_family="surface_code_rotated", distance=3))
    assert isinstance(out, CodeParamsOutput)
    assert out.ok is True
    assert (out.n, out.k, out.num_stabilizers) == (9, 1, 8)
    assert out.stim_code_distance == 3
    assert out.num_observables == 1
    assert out.notes == ""


def test_repetition_code_params():
    out = code_params(CodeParamsInput(code_family="repetition_code", distance=5))
    assert out.ok is True
    assert (out.n, out.k, out.num_stabilizers) == (5, 1, 4)
    assert out.stim_code_distance == 5


def test_color_code_reports_graphlike_lower_bound():
    out = code_params(CodeParamsInput(code_family="color_code", distance=3))
    assert out.ok is True
    assert (out.n, out.num_stabilizers) == (7, 6)
    assert out.stim_code_distance == 2  # graphlike distance < true distance 3
    assert out.notes != ""


def test_rejects_even_distance_for_surface():
    out = code_params(CodeParamsInput(code_family="surface_code_rotated", distance=4))
    assert out.ok is False
    assert "odd" in out.error_message.lower()
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_code_params.py -v` → FAIL (module not found).

- [ ] **Step 3: Write the implementation** — `src/chatqec_mcp_tools/tools/code_params.py`:
```python
"""code_params tool - report a named QEC code's parameters + a Stim-computed distance.

Reports [[n, k]] and the stabilizer-generator count from each family's standard
formula, plus the code distance computed by Stim (shortest_graphlike_error). No
qiskit-qec dependency. For color codes the graphlike distance is a lower bound on
the true distance (hyperedge errors are undercounted).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FAMILY_MAP = {
    "surface_code_rotated": "surface_code:rotated_memory_z",
    "repetition_code": "repetition_code:memory",
    "color_code": "color_code:memory_xyz",
}
_ODD_ONLY = {"surface_code_rotated", "color_code"}
_MIN_D, _MAX_D = 3, 15


def _n_physical(family: str, d: int) -> int:
    if family == "surface_code_rotated":
        return d * d
    if family == "repetition_code":
        return d
    return (3 * d * d + 1) // 4  # triangular 6.6.6 color code


class CodeParamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["surface_code_rotated", "repetition_code", "color_code"]
    distance: int = Field(..., ge=_MIN_D, le=_MAX_D)


class CodeParamsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    code_family: str = ""
    distance: int = 0
    n: int = 0
    k: int = 0
    num_stabilizers: int = 0
    stim_code_distance: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    notes: str = ""
    error_message: str = ""


def code_params(inp: CodeParamsInput) -> CodeParamsOutput:
    if inp.code_family in _ODD_ONLY and inp.distance % 2 == 0:
        return CodeParamsOutput(
            ok=False,
            error_message=f"{inp.code_family} requires odd distance; got {inp.distance}",
        )

    try:
        import stim  # noqa: PLC0415

        circuit = stim.Circuit.generated(
            _FAMILY_MAP[inp.code_family],
            distance=inp.distance,
            rounds=inp.distance,
            after_clifford_depolarization=0.001,
            before_measure_flip_probability=0.001,
            after_reset_flip_probability=0.001,
        )
        stim_distance = len(circuit.shortest_graphlike_error())
    except (ValueError, RuntimeError) as e:
        return CodeParamsOutput(ok=False, error_message=f"stim error: {e}")

    n = _n_physical(inp.code_family, inp.distance)
    k = 1
    notes = ""
    if inp.code_family == "color_code":
        notes = (
            "stim_code_distance is the graphlike distance (a lower bound); "
            "color-code hyperedge errors are undercounted"
        )
    return CodeParamsOutput(
        ok=True,
        code_family=inp.code_family,
        distance=inp.distance,
        n=n,
        k=k,
        num_stabilizers=n - k,
        stim_code_distance=stim_distance,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        notes=notes,
    )
```

- [ ] **Step 4: Register in `server.py`** — add the import:
```python
from chatqec_mcp_tools.tools.code_params import (
    CodeParamsInput,
    code_params,
)
```
add the handler:
```python
def _code_params_handler(args: dict[str, Any]) -> dict[str, Any]:
    return code_params(CodeParamsInput.model_validate(args)).model_dump()
```
append to `_TOOLS`:
```python
    ToolSpec(
        name="code_params",
        description=(
            "Report a named QEC code's parameters — [[n, k]], stabilizer-generator "
            "count, and the Stim-computed code distance — for a family "
            "(surface_code_rotated, repetition_code, color_code) at a given distance."
        ),
        input_schema=CodeParamsInput.model_json_schema(),
        handler=_code_params_handler,
    ),
```

- [ ] **Step 5: Update `README.md`** — add to the Tools list:
```markdown
- `code_params` — report a code's [[n,k]] + stabilizer count and a Stim-computed distance
```
and replace the `Deferred: qiskit_qec_run …` line with:
```markdown
Superseded: `qiskit_qec_run` (code construction via qiskit-qec) — replaced by the
Stim-based `code_params`, since qiskit-qec is unshippable (not on PyPI; abandoned,
broken against modern Qiskit). Future: `stabilizer_check`, `code_search`,
`logical_error_curve`, `decode_compare`, `monte_carlo_estimate`.
```

- [ ] **Step 6: Run tests + lint** — `uv run pytest -q` (all pass, incl. 4 new), `uv run ruff check .` (clean), `uv run mypy src` (clean).

- [ ] **Step 7: Commit**
```bash
cd ~/chatqec-mcp-tools
git add src/chatqec_mcp_tools/tools/code_params.py src/chatqec_mcp_tools/server.py README.md tests/test_code_params.py
git commit -m "feat(tools): add Stim-backed code_params tool (supersedes deferred qiskit_qec_run)"
```

---

### Task 2: main-repo wiring — prompt + Streamlit (main)

**Files:**
- Modify: `~/chatQEC/src/chatqec/conversation/prompts.py`
- Modify: `~/chatQEC/chatqec_app/components/tool_output.py`
- Modify: `~/chatQEC/tests/test_prompts_tools.py`

- [ ] **Step 1: Update the prompt test (RED)** — in `tests/test_prompts_tools.py`, add `"code_params"` to the asserted tuple:
```python
    for name in ("stim_simulate", "pymatching_decode", "threshold_sweep", "code_params"):
```
Run `cd ~/chatQEC && uv run pytest tests/test_prompts_tools.py -v` → FAIL (code_params not in prompt).

- [ ] **Step 2: Update the prompt** — in `src/chatqec/conversation/prompts.py`, extend the QEC-tools bullet. Change:
```
- To decode a circuit's syndromes and get its logical error rate, call
  `pymatching_decode`. To find or plot a code's threshold across physical error
  rates and distances, call `threshold_sweep`.
```
to:
```
- To decode a circuit's syndromes and get its logical error rate, call
  `pymatching_decode`. To find or plot a code's threshold across physical error
  rates and distances, call `threshold_sweep`. To report a code's parameters
  ([[n,k]], stabilizer count, Stim-computed distance), call `code_params`.
```
Run the prompt test → PASS.

- [ ] **Step 3: Render `code_params` in Streamlit** — in `chatqec_app/components/tool_output.py`, add `code_params` to the JSON-metric branch. Change:
```python
    if call.name == "pymatching_decode":
```
to:
```python
    if call.name in ("pymatching_decode", "code_params"):
```

- [ ] **Step 4: Verify** — `cd ~/chatQEC && uv run pytest tests/test_prompts_tools.py tests/test_streamlit_utils.py -q` (pass) and `uv run ruff check src/chatqec/conversation/prompts.py chatqec_app/components/tool_output.py tests/test_prompts_tools.py` (clean). Ensure pre-commit passes.

- [ ] **Step 5: Commit**
```bash
cd ~/chatQEC
git add src/chatqec/conversation/prompts.py chatqec_app/components/tool_output.py tests/test_prompts_tools.py
git commit -m "feat(tools): advertise + render code_params"
```

---

## Notes
- Task A is entirely in the sibling; Task B entirely in the main repo. The main app discovers `code_params` dynamically over stdio — Task B only edits prompt copy + rendering.
- All test assertion values were verified against real Stim output before this plan was written.
