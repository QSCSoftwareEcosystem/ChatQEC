"""ibm_add_pauli_checks tool -- add spacetime Pauli checks (IBM's qiskit-paulice) to a
Clifford circuit and report the cost: extra qubits, gates, depth, and, when a noise level
is given, the fraction of shots postselection would discard.

Needs the ``[paulice]`` extra (qiskit-paulice, qiskit, and qiskit-aer for the discard
estimate); everything is imported lazily so the server starts without it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware import _pauli_checks_impl as impl

_MAX_CIRCUIT_BYTES = 256 * 1024
_MAX_RETURNED_QASM_BYTES = 20_000  # keep a big checked circuit out of the LLM context
_MAX_QUBITS = 24  # the check search grows quickly with width; keeps calls inside the 60 s budget
_INSTALL_HINT = "uv tool install 'chatqec-mcp-tools[paulice]'"


class IbmAddPauliChecksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(..., description="A Clifford circuit that ends in measurements")
    circuit_format: Literal["qasm", "stim"] = Field(
        "qasm", description="OpenQASM 2 text, or Stim text (H, S, X, Y, Z, CX, CZ, SWAP, M)"
    )
    num_checks: int | None = Field(
        None,
        ge=0,
        le=64,
        description="Number of checks to add (default: the lowest-cost set the search finds)",
    )
    noise_p: float | None = Field(
        None,
        gt=0,
        le=0.1,
        description=(
            "Physical error rate. Used to choose checks (default 0.001) and, when given, to "
            "estimate the postselection discard rate by noisy simulation"
        ),
    )
    shots: int = Field(2_000, ge=1, le=50_000, description="Shots for the discard-rate estimate")
    seed: int | None = Field(
        default=None, description="Seed for the check search and the simulation"
    )


class PauliCheckOverhead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubits_before: int
    qubits_after: int
    gates_before: int = Field(description="Gate count, excluding measurements and barriers")
    gates_after: int
    depth_before: int
    depth_after: int


class IbmAddPauliChecksOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    circuit_qasm: str = ""
    overhead: PauliCheckOverhead | None = None
    num_checks: int = 0
    cost_gamma: float | None = Field(
        default=None, description="Paulice's sampling-overhead cost (>= 1)"
    )
    postselection_discard_rate: float | None = None
    notes: str = ""
    error_message: str = ""


def _fail(message: str) -> IbmAddPauliChecksOutput:
    return IbmAddPauliChecksOutput(ok=False, error_message=message)


def ibm_add_pauli_checks(inp: IbmAddPauliChecksInput) -> IbmAddPauliChecksOutput:
    if len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return _fail(f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)")
    try:
        circuit = impl.load_circuit(inp.circuit, inp.circuit_format)
        if circuit.num_qubits > _MAX_QUBITS:
            return _fail(
                f"circuit has {circuit.num_qubits} qubits; at most {_MAX_QUBITS} supported"
            )
        checked, note = impl.pick_checks(circuit, inp.num_checks, inp.noise_p, inp.seed)
        rate = (
            impl.discard_rate(checked, inp.noise_p, inp.shots, inp.seed)
            if inp.noise_p is not None
            else None
        )
        qasm = impl.to_qasm(checked.circuit)
    except ImportError:
        return _fail(f"qiskit-paulice is not installed: {_INSTALL_HINT}")
    except ValueError as exc:
        return _fail(str(exc))

    after = checked.circuit
    if len(qasm.encode("utf-8")) > _MAX_RETURNED_QASM_BYTES:
        omitted = (
            f"checked circuit omitted: {len(qasm) // 1024} KB of OpenQASM is too large to return"
        )
        note = f"{note} {omitted}".strip()
        qasm = ""
    return IbmAddPauliChecksOutput(
        ok=True,
        circuit_qasm=qasm,
        overhead=PauliCheckOverhead(
            qubits_before=circuit.num_qubits,
            qubits_after=after.num_qubits,
            gates_before=impl.count_gates(circuit),
            gates_after=impl.count_gates(after),
            depth_before=circuit.depth(),
            depth_after=after.depth(),
        ),
        num_checks=len(checked.check_qubits),
        cost_gamma=float(checked.cost),
        postselection_discard_rate=rate,
        notes=note,
    )
