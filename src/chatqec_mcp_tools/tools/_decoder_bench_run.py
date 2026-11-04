"""Run decoder-bench's own pipeline (vendored) on one memory configuration.

Mirrors decoder_bench/sampler.py at commit 9733a55: the circuit comes from the benchmark's
generator (SI1000 circuit noise, rounds = distance, Z basis), the detector error model is built
WITHOUT decomposition, check matrices come from its ``dem_to_check_matrices``, decoders are its
own classes with its parameters, and a shot fails when ``Decoder.verify`` says so. Only the
syndromes are sampled directly with Stim instead of being read back from an HDF5 dataset.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

DecoderBenchCode = Literal["surface", "color", "qldpc"]
# decoder-bench's registry names
DECODER_BENCH_DECODERS = ("pymatching", "belieffind", "bplsd", "relay-bp", "mwpf")
# Distances its generators support for circuit noise.
SUPPORTED_DISTANCES = {"surface": range(3, 16, 2), "color": range(3, 16, 2), "qldpc": (6, 10, 12)}
SOURCE = "decoder-bench (IEEE 11242095), commit 9733a55, vendored"


@dataclass(frozen=True)
class DecoderBenchResult:
    decoder: str
    ok: bool
    num_errors: int = 0
    shots: int = 0
    decode_seconds: float = 0.0
    note: str = ""


def build_circuit(code: DecoderBenchCode, distance: int, p: float) -> Any:
    from chatqec_mcp_tools.tools._vendor.decoder_bench import generators as g  # noqa: PLC0415

    if distance not in SUPPORTED_DISTANCES[code]:
        raise ValueError(
            f"decoder-bench {code} circuits support distance "
            f"{list(SUPPORTED_DISTANCES[code])}, not {distance}"
        )
    if code == "surface":
        return g.gen_surface_circuit((distance, p, "z"))
    if code == "color":
        return g.gen_color_circuit((distance, p))
    return g.gen_qldpc_circuit((distance, p, "z"))


def circuit_sha256(circuit: Any) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def run(
    circuit: Any, decoders: list[str], shots: int, seed: int | None
) -> list[DecoderBenchResult]:
    """Decode the same sampled syndromes with each decoder-bench decoder."""
    import numpy as np  # noqa: PLC0415

    from chatqec_mcp_tools.tools._vendor.decoder_bench import (  # noqa: PLC0415
        decoders as _registered,  # noqa: F401  (importing registers the decoder classes)
    )
    from chatqec_mcp_tools.tools._vendor.decoder_bench.build_circuit import (  # noqa: PLC0415
        dem_to_check_matrices,
    )
    from chatqec_mcp_tools.tools._vendor.decoder_bench.sampler_core import (  # noqa: PLC0415
        DecoderRegistry,
        DecoderState,
    )

    dem = circuit.detector_error_model()
    check_matrix, obs_matrix, priors = dem_to_check_matrices(dem)
    state = DecoderState(
        check_matrix=check_matrix, obs_matrix=obs_matrix, priors=priors, circuit=circuit, dem=dem
    )
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, bit_packed=False, separate_observables=True
    )
    results = []
    for name in decoders:
        try:
            decoder = DecoderRegistry.get(name)(state)
        except ImportError as exc:
            results.append(DecoderBenchResult(name, ok=False, note=f"not installed: {exc.name}"))
            continue
        start = time.perf_counter()
        predictions = decoder.decode_batch(dets)
        verified = decoder.verify_batch(predictions, obs)
        errors = int(len(verified) - int(np.sum(verified)))
        results.append(
            DecoderBenchResult(
                name,
                ok=True,
                num_errors=errors,
                shots=shots,
                decode_seconds=round(time.perf_counter() - start, 3),
            )
        )
    return results


def std_error(errors: int, shots: int) -> float:
    rate = errors / shots
    return math.sqrt(rate * (1 - rate) / shots)
