"""iqm_preview_execution tool -- what iqm_submit_execution would run, without submitting.
Free; call this before iqm_submit_execution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import compute_iqm_mapping

_MAX_NUM_QUBITS = 1024
_MAX_QAOA_LAYERS = 1000


class IqmPreviewExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = Field(default=None, ge=1, le=_MAX_NUM_QUBITS)
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    objective: str = "maximum_fidelity"
    shots: int = Field(default=1000, ge=1)


class IqmPreviewExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    dut_label: str = ""
    physical_qubits: list[str] = Field(default_factory=list)
    logical_to_physical: dict[str, str] = Field(default_factory=dict)
    strategy_used: str = ""
    estimated_fidelity: float = 0.0
    shots: int = 0
    backend_healthy: bool | None = None
    error_message: str = ""


def iqm_preview_execution(inp: IqmPreviewExecutionInput) -> IqmPreviewExecutionOutput:
    out = compute_iqm_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        objective=inp.objective,
        include_svg=False,
    )
    if not out.ok or out.placement is None:
        return IqmPreviewExecutionOutput(ok=False, error_message=out.error_message)
    l2p = out.physical_names()
    return IqmPreviewExecutionOutput(
        ok=True,
        backend=out.backend,
        dut_label=out.dut_label,
        physical_qubits=sorted(l2p.values(), key=lambda n: int(n[2:])),
        logical_to_physical={str(k): v for k, v in sorted(l2p.items())},
        strategy_used=out.placement.strategy_used,
        estimated_fidelity=out.estimated_fidelity,
        shots=inp.shots,
        backend_healthy=out.healthy,
    )


__all__ = ["IqmPreviewExecutionInput", "IqmPreviewExecutionOutput", "iqm_preview_execution"]
