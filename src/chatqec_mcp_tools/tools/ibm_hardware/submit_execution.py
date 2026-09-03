# src/chatqec_mcp_tools/tools/ibm_hardware/submit_execution.py
"""ibm_submit_execution tool -- SPENDS REAL CREDITS. Caps enforced before any
SDK call: num_qubits <= 10, shots <= 4000, backend required (no auto-pick).
Call ibm_preview_execution first.

Reuses compute_mapping's already-authenticated service/sdk_backend to build
the sampler instead of calling get_ibm_service()/service.backend() a second
time -- this tool has no IBM-auth code of its own."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import compute_mapping

_MAX_QUBITS = 10
_MAX_SHOTS = 4000
# Mirrors preview_execution.py's own _MAX_QAOA_LAYERS (module-private there
# too, so redefined rather than imported) -- compute_mapping's circuit
# construction is O(n * qaoa_layers), so an unbounded qaoa_layers is a
# cheap-to-request/expensive-to-build hazard independent of the credit-spend
# caps (_MAX_QUBITS/_MAX_SHOTS/required backend) this tool otherwise enforces
# at runtime before ever reaching compute_mapping/_make_sampler/sampler.run.
_MAX_QAOA_LAYERS = 1000


class IbmSubmitExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = None
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    backend: str = Field(..., min_length=1)
    shots: int


class IbmSubmitExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    job_id: str = ""
    backend: str = ""
    shots: int = 0
    status: str = ""
    error_message: str = ""


def _make_sampler(service: Any, backend: Any) -> Any:
    from qiskit_ibm_runtime import SamplerV2  # noqa: PLC0415

    return SamplerV2(mode=backend)


def ibm_submit_execution(inp: IbmSubmitExecutionInput) -> IbmSubmitExecutionOutput:
    if inp.shots > _MAX_SHOTS:
        return IbmSubmitExecutionOutput(
            ok=False, error_message=f"shots ({inp.shots}) exceeds the cap of {_MAX_SHOTS}."
        )
    if inp.num_qubits is not None and inp.num_qubits > _MAX_QUBITS:
        return IbmSubmitExecutionOutput(
            ok=False, error_message=f"num_qubits ({inp.num_qubits}) exceeds the cap of {_MAX_QUBITS}."
        )

    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective="maximum_fidelity",
        include_svg=False,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmSubmitExecutionOutput(ok=False, error_message=outcome.error_message)

    if len(outcome.placement.assignment) > _MAX_QUBITS:
        return IbmSubmitExecutionOutput(
            ok=False,
            error_message=(
                f"Circuit maps to {len(outcome.placement.assignment)} physical qubits, "
                f"which exceeds the cap of {_MAX_QUBITS}."
            ),
        )

    # Everything past this point is a real network operation (submitting the
    # job, then asking the returned job object for its id/status) and any of
    # it can fail in ways outside our control -- keep it all inside one try
    # block so nothing escapes ibm_submit_execution as a raised exception.
    try:
        sampler = _make_sampler(outcome.service, outcome.sdk_backend)
        job = sampler.run([], shots=inp.shots)
        job_id = job.job_id()
        status = str(job.status())
    except Exception as exc:
        return IbmSubmitExecutionOutput(ok=False, error_message=f"Submission failed: {exc}")

    return IbmSubmitExecutionOutput(
        ok=True,
        job_id=job_id,
        backend=inp.backend,
        shots=inp.shots,
        status=status,
    )


__all__ = [
    "IbmSubmitExecutionInput",
    "IbmSubmitExecutionOutput",
    "ibm_submit_execution",
]
