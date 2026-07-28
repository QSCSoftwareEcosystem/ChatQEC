"""tsim_simulate tool - run a Tsim circuit (Stim-compatible + non-Clifford
gates: T, T_DAG, R_X/R_Y/R_Z, U3, TPP/TPP_DAG, R_XX/R_YY/R_ZZ/R_PAULI, CCZ,
CCX) and return measurement/detector samples.

Only this file imports tsim.* — see docs/planning's QuEra integration plan
for why (isolates upstream churn to one seam, same pattern as
lightstim_adapter.py in this repo).

API verified against the installed `bloqade-tsim==0.1.5` package (not just
its README): `tsim.Circuit.compile_detector_sampler(*, strategy="cat5",
seed=None) -> CompiledDetectorSampler` takes no reference-sample kwargs.
`CompiledDetectorSampler.sample(shots, *, ..., use_detector_reference_sample=
False, use_observable_reference_sample=False, ...)` is where those two
kwargs actually live — confirmed via `inspect.signature` on the installed
package and cross-checked against
https://tsim.mintlify.app/api-reference/sampler.md.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_MAX_CIRCUIT_BYTES = 256 * 1024  # 256 KB
_MAX_SHOTS = 100_000  # lower than stim_simulate's 1M - Tsim's ZX/JAX pipeline
                       # is far more expensive per shot for non-Clifford circuits


class TsimSimulateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(
        ..., description="Tsim circuit source (Stim syntax + T/T_DAG/R_X/R_Y/"
        "R_Z/U3/TPP/TPP_DAG/R_XX/R_YY/R_ZZ/R_PAULI/CCZ/CCX)"
    )
    shots: int = Field(1, ge=1, description="Number of shots to sample")
    use_reference_sample: bool = Field(
        False,
        description="If true, XOR against a noiseless reference sample so "
        "detector/observable bits mean 'deviation from noiseless', matching "
        "Stim's convention. If false (default), Tsim's own raw convention: "
        "actual detector outcomes, not XORed flips.",
    )


class TsimSimulateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    shots: int = 0
    summary: str = ""
    samples_preview: list[list[int]] = Field(default_factory=list)
    error_message: str = ""


def tsim_simulate(inp: TsimSimulateInput) -> TsimSimulateOutput:
    if len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return TsimSimulateOutput(
            ok=False,
            error_message=f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)",
        )
    if inp.shots > _MAX_SHOTS:
        return TsimSimulateOutput(
            ok=False, error_message=f"shots > {_MAX_SHOTS:,} not permitted"
        )

    try:
        import tsim  # noqa: PLC0415
    except ImportError as e:
        return TsimSimulateOutput(ok=False, error_message=f"tsim not installed: {e}")

    try:
        circuit = tsim.Circuit(inp.circuit)
        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(
            shots=inp.shots,
            use_detector_reference_sample=inp.use_reference_sample,
            use_observable_reference_sample=inp.use_reference_sample,
        )
    except (ValueError, RuntimeError) as e:
        return TsimSimulateOutput(ok=False, error_message=f"tsim error: {e}")

    preview_rows = min(samples.shape[0], 20)
    preview = [list(map(int, samples[i].tolist())) for i in range(preview_rows)]
    truncated = preview_rows < samples.shape[0]
    summary = (
        f"shape={samples.shape}, dtype={samples.dtype}; "
        f"showing first {preview_rows} of {samples.shape[0]} rows"
        + (
            " (TRUNCATED — remaining rows were sampled but are not shown; "
            "do not state or assume their values)"
            if truncated
            else " (ALL rows shown — none omitted)"
        )
    )
    return TsimSimulateOutput(
        ok=True, shots=inp.shots, summary=summary, samples_preview=preview
    )
