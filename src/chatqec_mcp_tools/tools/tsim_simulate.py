"""tsim_simulate tool - run a Tsim circuit (Stim-compatible + non-Clifford
gates: T, T_DAG, R_X/R_Y/R_Z, U3, TPP/TPP_DAG, R_XX/R_YY/R_ZZ/R_PAULI, CCZ,
CCX) and return detector/observable samples (NOT raw measurement bits --
`compile_detector_sampler()` only ever samples DETECTOR/OBSERVABLE_INCLUDE
annotations, never bare `M` results; see the no-detectors-or-observables
guard below).

Only this file imports tsim.* — see docs/planning's QuEra integration plan
for why (isolates upstream churn to one seam, same pattern as
lightstim_adapter.py in this repo).

API verified against the installed `bloqade-tsim==0.1.5` package (not just
its README): `tsim.Circuit.compile_detector_sampler(*, strategy="cat5",
seed=None) -> CompiledDetectorSampler` takes no reference-sample kwargs.
`CompiledDetectorSampler.sample(shots, *, ..., append_observables=False,
use_detector_reference_sample=False, use_observable_reference_sample=False,
...)` is where those kwargs actually live — confirmed via
`inspect.signature` on the installed package and cross-checked against
https://tsim.mintlify.app/api-reference/sampler.md. Notably,
`use_observable_reference_sample` only affects observable *bits*; those bits
are excluded from the returned array entirely unless `append_observables=True`
is also passed (confirmed empirically: an `OBSERVABLE_INCLUDE`-only circuit
samples as shape `(shots, 0)` without it) -- this tool always passes
`append_observables=True` so that field is not a no-op.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import tsim

_MAX_CIRCUIT_BYTES = 256 * 1024  # 256 KB
_MAX_SHOTS = 100_000  # Lower than stim_simulate's 1M, but NOT because Tsim is
# more expensive per shot -- measured shots vectorize almost for free under
# JAX (100 -> 100,000 shots measured ~1.24s -> ~1.33s for the same circuit,
# see the fix-wave task report for the raw numbers). Kept as a conservative
# parity ceiling with the rest of this codebase; the real cost driver for
# Tsim is non-Clifford gate count, bounded separately by _MAX_TCOUNT below.

_MAX_TCOUNT = 18  # Ceiling on tsim.Circuit.tcount() -- the number of T-gates
# (T-gate-equivalent, after REPEAT-block expansion and decomposition-weighting
# non-T non-Clifford gates: e.g. CCZ/CCX ~7, U3 ~3, single T/T_DAG/R_*/TPP ~1)
# in the circuit. This is what drives Tsim's cost exponentially, not shots
# (see _MAX_SHOTS above) and not schema-valid circuit-text size. Measured on
# the installed bloqade-tsim==0.1.5 (single-qubit T-rotation microbenchmark,
# see the fix-wave task report): tcount=16 -> ~4.3s, tcount=18 -> ~5.3s,
# tcount=20 -> ~12.6s, tcount=24 -> ~75s, tcount=26 -> did not finish in 17
# minutes (killed). `tcount()` itself is cheap regardless of circuit
# complexity (sub-5ms even at tcount=40 in the same benchmark), so it is
# safe to call before the expensive `compile_detector_sampler()` step. 18
# keeps worst-observed latency at roughly 5s.


class TsimSimulateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(
        ..., description="Tsim circuit source (Stim syntax + T/T_DAG/R_X/R_Y/"
        "R_Z/U3/TPP/TPP_DAG/R_XX/R_YY/R_ZZ/R_PAULI/CCZ/CCX). Must contain at "
        "least one DETECTOR or OBSERVABLE_INCLUDE instruction -- this tool "
        "returns detector/observable samples, not raw measurement bits."
    )
    shots: int = Field(1, ge=1, description="Number of shots to sample")
    use_reference_sample: bool = Field(
        False,
        description="If true, XOR against a noiseless reference sample so "
        "detector/observable bits mean 'deviation from noiseless', matching "
        "Stim's convention. If false (default), Tsim's own raw convention: "
        "actual detector/observable outcomes, not XORed flips.",
    )


class TsimSimulateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    shots: int = 0
    summary: str = ""
    samples_preview: list[list[int]] = Field(default_factory=list)
    error_message: str = ""


def _pre_sample_check(circuit: tsim.Circuit) -> str | None:
    """Cheap validation run before the expensive compile_detector_sampler()
    step. Returns an error message, or None if the circuit is safe to sample.
    """
    tcount = circuit.tcount()
    if tcount > _MAX_TCOUNT:
        return (
            f"circuit has {tcount} non-Clifford T-gate-equivalents "
            f"(tcount) > {_MAX_TCOUNT} cap; reduce non-Clifford gate "
            "count (T/T_DAG/TPP/TPP_DAG/R_*/U3/CCZ/CCX)"
        )
    if circuit.num_detectors == 0 and circuit.num_observables == 0:
        return (
            "circuit has no DETECTOR or OBSERVABLE_INCLUDE instructions to "
            "sample; tsim_simulate returns detector/observable samples, not "
            "raw measurement bits -- add DETECTOR and/or OBSERVABLE_INCLUDE "
            "annotations referencing your M results"
        )
    return None


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
    except (ImportError, RuntimeError, AttributeError) as e:
        # `tsim` transitively imports jax/jaxlib/equinox; an import-time
        # jaxlib/CUDA mismatch on the host can surface as RuntimeError or
        # AttributeError rather than a clean ImportError, so all three are
        # treated as "tsim not usable" here rather than letting the rarer
        # ones escape uncaught.
        return TsimSimulateOutput(ok=False, error_message=f"tsim not installed: {e}")

    try:
        circuit = tsim.Circuit(inp.circuit)

        error_message = _pre_sample_check(circuit)
        if error_message is not None:
            return TsimSimulateOutput(ok=False, error_message=error_message)

        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(
            shots=inp.shots,
            append_observables=True,
            use_detector_reference_sample=inp.use_reference_sample,
            use_observable_reference_sample=inp.use_reference_sample,
        )
    except (ValueError, RuntimeError, IndexError, KeyError, TypeError, AttributeError) as e:
        return TsimSimulateOutput(ok=False, error_message=f"tsim error: {e}")

    preview_rows = min(samples.shape[0], 20)
    preview = [list(map(int, samples[i].tolist())) for i in range(preview_rows)]
    truncated = preview_rows < samples.shape[0]
    summary = (
        f"shape={samples.shape} (detector+observable columns: "
        f"{circuit.num_detectors} detectors, {circuit.num_observables} observables), "
        f"dtype={samples.dtype}; showing first {preview_rows} of {samples.shape[0]} rows"
        + (
            " (TRUNCATED — remaining rows were sampled but are not shown; "
            "do not state or assume their values)"
            if truncated
            else " (ALL rows shown — none omitted)"
        )
    )
    return TsimSimulateOutput(
        ok=True, shots=samples.shape[0], summary=summary, samples_preview=preview
    )
