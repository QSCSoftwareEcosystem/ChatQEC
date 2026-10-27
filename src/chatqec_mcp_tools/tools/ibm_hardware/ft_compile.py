"""ibm_ft_compile tool -- compile a circuit for fault-tolerant execution with Qiskit's
preset pass managers (Qiskit >= 2.5, the ``[ft]`` extra).

Two modes:
  - ``clifford_t``: rotations are approximated with gridsynth (Ross-Selinger) into Clifford+T;
    the cost to watch is the T count.
  - ``pbc``: Pauli-based computation (Litinski transformation): the circuit becomes a
    sequence of Pauli product rotations; the cost to watch is the rotation count.

Rotation precision matters: Qiskit only honours an explicit error budget when BOTH
``rz_synthesis_error`` and ``rz_cache_error`` are set (else it derives them from
``approximation_degree``), so both are passed. Measured on ``rz(0.3)``: 2, 4, 6, 10 digits
give 20, 42, 65, 101 T gates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

_MAX_CIRCUIT_BYTES = 256 * 1024
_MAX_QUBITS = 64
_MAX_RETURNED_QASM_BYTES = (
    20_000  # a gridsynth circuit can have 10^4+ gates; keep it out of the LLM context
)
_CACHE_ERROR_FRACTION = 0.1  # cache tolerance as a fraction of the synthesis error
_T_GATES = ("t", "tdg")
_CLIFFORD_GATES = frozenset(
    {"h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "cx", "cy", "cz", "swap", "id", "ecr"}
)
_NON_GATES = frozenset({"measure", "barrier"})
_INSTALL_HINT = (
    "Qiskit >= 2.5 is required for fault-tolerant compilation: "
    "uv tool install 'chatqec-mcp-tools[ft]'"
)


class IbmFtCompileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str = Field(..., description="OpenQASM 2 circuit")
    mode: Literal["clifford_t", "pbc"] = Field(
        "clifford_t",
        description="clifford_t: Clifford+T with gridsynth; pbc: Pauli-based computation",
    )
    gridsynth_precision: int = Field(
        10,
        ge=1,
        le=60,
        description=(
            "Decimal digits of rotation precision (clifford_t): rotations are approximated "
            "to 10**-precision. Higher means more T gates"
        ),
    )
    seed: int | None = Field(
        default=None, description="Seed for the stochastic parts of the transpiler"
    )


class IbmFtCompileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    mode: str = ""
    t_count: int | None = Field(default=None, description="T and T-dagger gates (clifford_t only)")
    clifford_count: int | None = Field(default=None, description="Clifford gates (clifford_t only)")
    num_rotations: int | None = Field(
        default=None, description="Pauli product rotations (pbc only)"
    )
    depth: int = 0
    num_qubits: int = 0
    circuit_qasm: str = Field(
        default="", description="Compiled circuit; empty when too large to return"
    )
    notes: str = ""
    error_message: str = ""


def _fail(message: str) -> IbmFtCompileOutput:
    return IbmFtCompileOutput(ok=False, error_message=message)


def _to_qasm(circuit: object) -> str:
    from qiskit import qasm2, qasm3  # noqa: PLC0415

    try:
        return str(qasm2.dumps(circuit))
    except Exception:
        return str(qasm3.dumps(circuit))


def ibm_ft_compile(inp: IbmFtCompileInput) -> IbmFtCompileOutput:
    if len(inp.circuit_qasm.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return _fail(f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)")
    try:
        from qiskit.transpiler import (  # noqa: PLC0415
            generate_preset_clifford_t_pass_manager,
            generate_preset_pbc_pass_manager,
        )

        circuit = parse_circuit_input(
            circuit_qasm=inp.circuit_qasm, circuit_preset=None, num_qubits=None, qaoa_layers=None
        ).circuit
        if circuit.num_qubits > _MAX_QUBITS:
            return _fail(
                f"circuit has {circuit.num_qubits} qubits; at most {_MAX_QUBITS} supported"
            )
        if inp.mode == "pbc":
            compiled = generate_preset_pbc_pass_manager(seed_transpiler=inp.seed).run(circuit)
        else:
            eps = 10.0**-inp.gridsynth_precision
            compiled = generate_preset_clifford_t_pass_manager(
                seed_transpiler=inp.seed,
                rz_synthesis_config={
                    "rz_synthesis_error": eps,
                    "rz_cache_error": eps * _CACHE_ERROR_FRACTION,
                },
            ).run(circuit)
    except ImportError:
        return _fail(_INSTALL_HINT)
    except ValueError as exc:
        return _fail(str(exc))
    except Exception as exc:  # a Qiskit transpiler failure on an unsupported gate, etc.
        return _fail(f"compilation failed: {type(exc).__name__}: {exc}")

    ops = compiled.count_ops()
    is_pbc = inp.mode == "pbc"
    qasm = _to_qasm(compiled)
    notes = ""
    if len(qasm.encode("utf-8")) > _MAX_RETURNED_QASM_BYTES:
        notes = (
            f"compiled circuit omitted: {len(qasm) // 1024} KB of OpenQASM is too large to "
            "return; the counts above describe it"
        )
        qasm = ""
    return IbmFtCompileOutput(
        ok=True,
        mode=inp.mode,
        t_count=None if is_pbc else sum(ops.get(g, 0) for g in _T_GATES),
        clifford_count=None if is_pbc else sum(ops.get(g, 0) for g in _CLIFFORD_GATES),
        num_rotations=ops.get("pauli_product_rotation", 0) if is_pbc else None,
        depth=compiled.depth(),
        num_qubits=compiled.num_qubits,
        circuit_qasm=qasm,
        notes=notes,
    )
