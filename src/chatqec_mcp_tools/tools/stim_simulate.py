"""stim_simulate tool - run a Stim circuit and return measurement samples."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_MAX_CIRCUIT_BYTES = 256 * 1024  # 256 KB
_MAX_SHOTS = 1_000_000


class StimSimulateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(..., description="Stim circuit source code")
    shots: int = Field(1, ge=1, description="Number of shots to sample")


class StimSimulateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    shots: int
    summary: str
    samples_preview: list[list[int]] = Field(default_factory=list)
    error_message: str = ""


def stim_simulate(inp: StimSimulateInput) -> StimSimulateOutput:
    if len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return StimSimulateOutput(
            ok=False,
            shots=0,
            summary="",
            error_message=f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)",
        )
    if inp.shots > _MAX_SHOTS:
        return StimSimulateOutput(
            ok=False,
            shots=0,
            summary="",
            error_message=f"shots > {_MAX_SHOTS:,} not permitted",
        )

    try:
        import stim  # noqa: PLC0415

        circuit = stim.Circuit(inp.circuit)
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=inp.shots)
    except (ValueError, RuntimeError) as e:
        return StimSimulateOutput(
            ok=False, shots=0, summary="", error_message=f"stim error: {e}",
        )

    preview_cap = min(20, _MAX_SHOTS)
    preview_rows = min(samples.shape[0], preview_cap)
    preview = [list(map(int, samples[i].tolist())) for i in range(preview_rows)]
    truncated = preview_rows < samples.shape[0]
    summary = (
        f"shape={samples.shape}, dtype={samples.dtype}; "
        f"showing first {preview_rows} of {samples.shape[0]} rows"
        + (" (TRUNCATED — remaining rows were sampled but are not shown; "
           "do not state or assume their values)"
           if truncated else " (ALL rows shown — none omitted)")
    )
    return StimSimulateOutput(
        ok=True, shots=inp.shots, summary=summary, samples_preview=preview,
    )
