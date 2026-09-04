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
    # Deliberately ge=1 only, no upper bound: the real cap (_MAX_QUBITS) is a
    # runtime ok=False check below, not a pydantic ValidationError, so that a
    # too-large request still returns a structured error instead of raising.
    # ge=1 alone still blocks num_qubits <= 0, which reaches qiskit's own
    # CircuitError with no cap-check in between.
    num_qubits: int | None = Field(default=None, ge=1)
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    backend: str = Field(..., min_length=1)
    # Same reasoning as num_qubits above: ge=1 only, no upper bound -- the
    # _MAX_SHOTS cap is enforced at runtime as ok=False.
    shots: int = Field(..., ge=1)


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


def _transpile(circuit: Any, backend: Any, initial_layout: list[int]) -> Any:
    """Lazy wrapper around qiskit.transpile -- same lazy-import convention as
    _make_sampler/_client.get_ibm_service (this whole package avoids importing
    qiskit at module scope so it stays importable without the optional 'ibm'
    extra); kept as its own function so tests can patch it directly, exactly
    like _make_sampler already is."""
    from qiskit import transpile  # noqa: PLC0415

    return transpile(circuit, backend=backend, initial_layout=initial_layout)


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

    # Build the actual ISA circuit to run: compute_mapping only produces a
    # placement (logical -> physical qubit assignment), not something
    # runnable. The mapped circuit itself (outcome.circuit) typically has no
    # measurements (the mapping/preview tools never need them), so add them
    # here if missing -- a real hardware run needs measurements to get any
    # data back. transpile() then places qubits per the resolved placement.
    try:
        circuit = outcome.circuit
        if not circuit.clbits:
            circuit = circuit.copy()
            circuit.measure_all()

        initial_layout = [outcome.placement.assignment[i] for i in range(circuit.num_qubits)]
        isa_circuit = _transpile(circuit, outcome.sdk_backend, initial_layout)

        sampler = _make_sampler(outcome.service, outcome.sdk_backend)
        job = sampler.run([isa_circuit], shots=inp.shots)
    except Exception as exc:
        return IbmSubmitExecutionOutput(
            ok=False,
            error_message=(
                f"Submission failed: {exc}. A job may still have been created on the "
                "backend -- check your IBM Quantum jobs list before retrying."
            ),
        )

    # job.job_id() and job.status() are separate calls on the now-real,
    # already-submitted job: a failure in either must not discard the job_id
    # we already have (the job is real and spending credits regardless), and
    # must not raise out of this function.
    try:
        job_id = job.job_id()
    except Exception:
        job_id = ""
    try:
        status = str(job.status())
    except Exception as exc:
        status = f"unknown (status lookup failed: {exc})"

    return IbmSubmitExecutionOutput(
        ok=True,
        job_id=job_id,
        backend=outcome.backend,  # the resolved backend name, not the raw request
        shots=inp.shots,
        status=status,
    )


__all__ = [
    "IbmSubmitExecutionInput",
    "IbmSubmitExecutionOutput",
    "ibm_submit_execution",
]
