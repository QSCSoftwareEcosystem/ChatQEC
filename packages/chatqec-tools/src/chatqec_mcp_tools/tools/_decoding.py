"""Shared Stim + PyMatching decode primitive.

Used by both the `pymatching_decode` tool (decode one circuit) and the
`threshold_sweep` tool (decode many generated circuits). Kept in one place so the
detector-error-model / matching / batch-decode logic is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stim


@dataclass(frozen=True)
class DecodeStats:
    logical_error_rate: float
    num_errors: int
    shots: int
    num_detectors: int
    num_observables: int


def decode_circuit(circuit: stim.Circuit, shots: int) -> DecodeStats:
    """Decode a Stim circuit's syndromes with MWPM and return logical-error stats.

    Raises:
        ValueError: if the circuit defines no logical observables to decode.
    """
    import pymatching  # noqa: PLC0415

    dem = circuit.detector_error_model(decompose_errors=True)
    if dem.num_observables == 0:
        raise ValueError("circuit defines no logical observables to decode")

    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(shots=shots, separate_observables=True)
    predictions = matching.decode_batch(detections)

    num_errors = int((predictions != observables).any(axis=1).sum())
    return DecodeStats(
        logical_error_rate=num_errors / shots if shots else 0.0,
        num_errors=num_errors,
        shots=shots,
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )
