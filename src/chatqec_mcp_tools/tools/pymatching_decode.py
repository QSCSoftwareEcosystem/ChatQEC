"""pymatching_decode tool - MWPM-decode a Stim circuit; report logical error rate."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._decoding import decode_circuit

_MAX_CIRCUIT_BYTES = 256 * 1024
_MAX_SHOTS = 1_000_000


class PymatchingDecodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(..., description="Stim circuit with DETECTORs and OBSERVABLEs")
    shots: int = Field(1, ge=1, description="Number of syndrome shots to decode")


class PymatchingDecodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    logical_error_rate: float = 0.0
    num_errors: int = 0
    shots: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    error_message: str = ""


def pymatching_decode(inp: PymatchingDecodeInput) -> PymatchingDecodeOutput:
    if len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return PymatchingDecodeOutput(
            ok=False, error_message=f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)"
        )
    if inp.shots > _MAX_SHOTS:
        return PymatchingDecodeOutput(
            ok=False, error_message=f"shots > {_MAX_SHOTS:,} not permitted"
        )

    try:
        import stim  # noqa: PLC0415

        circuit = stim.Circuit(inp.circuit)
        stats = decode_circuit(circuit, inp.shots)
    except (ValueError, RuntimeError) as e:
        return PymatchingDecodeOutput(ok=False, error_message=f"decode error: {e}")

    return PymatchingDecodeOutput(
        ok=True,
        logical_error_rate=stats.logical_error_rate,
        num_errors=stats.num_errors,
        shots=stats.shots,
        num_detectors=stats.num_detectors,
        num_observables=stats.num_observables,
    )
