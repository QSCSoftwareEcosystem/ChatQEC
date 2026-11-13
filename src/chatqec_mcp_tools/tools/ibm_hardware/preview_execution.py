# src/chatqec_mcp_tools/tools/ibm_hardware/preview_execution.py
"""ibm_preview_execution tool -- everything ibm_submit_execution would do,
minus submission. Free; call this before ibm_submit_execution.

Reuses compute_mapping's already-authenticated sdk_backend for the status
check instead of calling get_ibm_service()/service.backend() a second time --
this tool has no IBM-auth code of its own."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import Mapper, compute_mapping

# Mirrors map_circuit.py's _MAX_NUM_QUBITS (that name is module-private and not
# exported, so it's redefined here rather than imported) -- same reasoning
# applies to qaoa_layers: compute_mapping's circuit construction is O(n *
# qaoa_layers), so an unbounded qaoa_layers is just as much of a
# cheap-to-request/expensive-to-build hazard as an unbounded num_qubits.
_MAX_NUM_QUBITS = 1024
_MAX_QAOA_LAYERS = 1000


class IbmPreviewExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = Field(default=None, ge=1, le=_MAX_NUM_QUBITS)
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    backend: str
    objective: str = "maximum_fidelity"
    mapper: Mapper = "default"
    shots: int = 1000


class IbmPreviewExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    physical_qubits: list[int] = Field(default_factory=list)
    strategy_used: str = ""
    two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    shots: int = 0
    estimated_queue_position: int | None = None
    backend_operational: bool = False
    error_message: str = ""


def ibm_preview_execution(inp: IbmPreviewExecutionInput) -> IbmPreviewExecutionOutput:
    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective=inp.objective,
        include_svg=False,
        mapper=inp.mapper,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmPreviewExecutionOutput(ok=False, error_message=outcome.error_message)

    try:
        status = outcome.sdk_backend.status()
        operational, pending_jobs = status.operational, status.pending_jobs
    except Exception as exc:
        return IbmPreviewExecutionOutput(
            ok=False, error_message=f"Failed to check backend status: {exc}"
        )

    return IbmPreviewExecutionOutput(
        ok=True,
        backend=outcome.backend,
        physical_qubits=sorted(outcome.routed_layout or outcome.placement.assignment.values()),
        strategy_used=outcome.placement.strategy_used,
        two_qubit_gate_count=outcome.two_qubit_gate_count,
        estimated_fidelity=outcome.estimated_fidelity,
        shots=inp.shots,
        estimated_queue_position=pending_jobs,
        backend_operational=operational,
    )


__all__ = [
    "IbmPreviewExecutionInput",
    "IbmPreviewExecutionOutput",
    "ibm_preview_execution",
]
