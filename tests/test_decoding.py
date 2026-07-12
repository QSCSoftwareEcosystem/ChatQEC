"""Shared Stim+PyMatching decode primitive tests."""

import stim

from chatqec_mcp_tools.tools._decoding import DecodeStats, decode_circuit


def _rep_code(distance=3, rounds=3, p=0.05):
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


def test_decode_circuit_returns_stats_in_range():
    stats = decode_circuit(_rep_code(), shots=200)
    assert isinstance(stats, DecodeStats)
    assert stats.shots == 200
    assert 0.0 <= stats.logical_error_rate <= 0.5
    assert stats.num_observables >= 1
    assert stats.num_detectors >= 1
    assert stats.num_errors == round(stats.logical_error_rate * 200)


def test_decode_circuit_raises_without_observables():
    # A bare measurement circuit has no observables/detectors.
    circuit = stim.Circuit("M 0\n")
    try:
        decode_circuit(circuit, shots=4)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
