"""iqm_submit_execution tool -- SUBMITS A REAL JOB to the IQM quantum computer.

Caps are checked before any network call: at most 10 mapped qubits and 4000 shots, and
``confirm_submit`` must be true. The circuit is placed by compute_iqm_mapping (same engine
as ibm_map_circuit), then the IQM worker transpiles it to the device's native gates
(r, cz) with that placement as the initial layout and submits it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError, call_worker
from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import compute_iqm_mapping

_MAX_QUBITS = 10
_MAX_SHOTS = 4000
_MAX_QAOA_LAYERS = 1000


class IqmSubmitExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    # ge=1 only: the real caps are runtime ok=False checks, so an oversized request gets a
    # structured refusal rather than a validation error.
    num_qubits: int | None = Field(default=None, ge=1)
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    shots: int = Field(..., ge=1)
    use_timeslot: bool = False
    confirm_submit: bool = Field(
        default=False,
        description="Must be true: the user has seen iqm_preview_execution and wants to run.",
    )


class IqmSubmitExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    job_id: str = ""
    backend: str = ""
    physical_qubits: list[str] = Field(default_factory=list)
    shots: int = 0
    status: str = ""
    transpiled_two_qubit_gate_count: int = 0
    transpiled_depth: int = 0
    error_message: str = ""


def iqm_submit_execution(  # noqa: PLR0911 -- distinct refusal paths
    inp: IqmSubmitExecutionInput,
) -> IqmSubmitExecutionOutput:
    if not inp.confirm_submit:
        return IqmSubmitExecutionOutput(
            ok=False,
            error_message=(
                "Not submitted: confirm_submit must be true. Show the user "
                "iqm_preview_execution first and submit only after they agree."
            ),
        )
    if inp.shots > _MAX_SHOTS:
        return IqmSubmitExecutionOutput(
            ok=False, error_message=f"shots ({inp.shots}) exceeds the cap of {_MAX_SHOTS}."
        )
    if inp.num_qubits is not None and inp.num_qubits > _MAX_QUBITS:
        return IqmSubmitExecutionOutput(
            ok=False,
            error_message=f"num_qubits ({inp.num_qubits}) exceeds the cap of {_MAX_QUBITS}.",
        )

    out = compute_iqm_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        objective="maximum_fidelity",
        include_svg=False,
    )
    if not out.ok or out.placement is None:
        return IqmSubmitExecutionOutput(ok=False, error_message=out.error_message)
    if not out.placement.assignment:
        return IqmSubmitExecutionOutput(
            ok=False, error_message="The placement is empty; refusing to submit nothing."
        )
    if len(out.placement.assignment) > _MAX_QUBITS:
        return IqmSubmitExecutionOutput(
            ok=False,
            error_message=(
                f"Circuit maps to {len(out.placement.assignment)} physical qubits, "
                f"which exceeds the cap of {_MAX_QUBITS}."
            ),
        )
    if out.healthy is False:
        return IqmSubmitExecutionOutput(
            ok=False, error_message="The IQM server reports the device as unhealthy."
        )

    try:
        from qiskit import qasm2  # noqa: PLC0415

        circuit = out.circuit
        if not circuit.clbits:  # hardware needs measurements to return any data
            circuit = circuit.copy()
            circuit.measure_all()
        names = out.physical_names()
        layout = [names[i] for i in range(circuit.num_qubits)]
        reply = call_worker(
            "submit",
            qasm=qasm2.dumps(circuit),
            initial_layout=layout,
            shots=inp.shots,
            use_timeslot=inp.use_timeslot,
        )
    except IQMCredentialsError as exc:
        return IqmSubmitExecutionOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IqmSubmitExecutionOutput(
            ok=False,
            error_message=(
                f"Submission failed: {exc}. A job may still have been created; check "
                "your IQM job list before retrying."
            ),
        )

    return IqmSubmitExecutionOutput(
        ok=True,
        job_id=reply.get("job_id", ""),
        backend=out.backend,
        physical_qubits=layout,
        shots=inp.shots,
        status=reply.get("status", ""),
        transpiled_two_qubit_gate_count=int(reply.get("transpiled_two_qubit_gate_count", 0)),
        transpiled_depth=int(reply.get("transpiled_depth", 0)),
    )


__all__ = ["IqmSubmitExecutionInput", "IqmSubmitExecutionOutput", "iqm_submit_execution"]
