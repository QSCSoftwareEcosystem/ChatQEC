"""pymatching_decode tool tests."""

import stim

from chatqec_mcp_tools.tools.pymatching_decode import (
    PymatchingDecodeInput,
    PymatchingDecodeOutput,
    pymatching_decode,
)

_REP = str(stim.Circuit.generated(
    "repetition_code:memory", distance=3, rounds=3,
    before_round_data_depolarization=0.05,
    before_measure_flip_probability=0.05,
))


def test_decode_reports_logical_error_rate():
    out = pymatching_decode(PymatchingDecodeInput(circuit=_REP, shots=200))
    assert isinstance(out, PymatchingDecodeOutput)
    assert out.ok is True
    assert out.shots == 200
    assert 0.0 <= out.logical_error_rate <= 0.5
    assert out.num_observables >= 1


def test_decode_errors_on_circuit_without_observables():
    out = pymatching_decode(PymatchingDecodeInput(circuit="M 0\n", shots=8))
    assert out.ok is False
    assert "observable" in out.error_message.lower()


def test_decode_rejects_oversize_circuit():
    huge = "M 0\n" * 100_000  # > 256 KB
    out = pymatching_decode(PymatchingDecodeInput(circuit=huge, shots=1))
    assert out.ok is False
    assert "too large" in out.error_message.lower()


def test_decode_reports_error_for_invalid_circuit():
    out = pymatching_decode(PymatchingDecodeInput(circuit="NOT_A_GATE 0\n", shots=4))
    assert out.ok is False
    assert out.error_message != ""


def test_decode_caps_shots():
    out = pymatching_decode(PymatchingDecodeInput(circuit=_REP, shots=10_000_000))
    assert out.ok is False
    assert "shots" in out.error_message.lower()
