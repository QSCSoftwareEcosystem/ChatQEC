"""decoder_compare: same syndromes through several decoders, ranked by LER."""

import builtins

import pytest
import stim
from pydantic import ValidationError

from chatqec_mcp_tools.tools.decoder_compare import DecoderCompareInput, decoder_compare

_HAS_BLOQADE = True
try:
    import bloqade.decoders  # noqa: F401
except ImportError:
    _HAS_BLOQADE = False
needs_decoders = pytest.mark.skipif(not _HAS_BLOQADE, reason="decoders extra not installed")


def _circuit(task: str = "repetition_code:memory", d: int = 3, p: float = 0.01) -> str:
    return str(
        stim.Circuit.generated(
            task,
            distance=d,
            rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
        )
    )


def test_pymatching_baseline_works_on_base_install():
    out = decoder_compare(
        DecoderCompareInput(circuit=_circuit(), decoders=["pymatching"], shots=500, seed=1)
    )
    assert out.ok and out.best == "pymatching"
    (r,) = out.results
    assert r.shots == 500 and 0 <= r.logical_error_rate < 0.2
    assert r.std_error >= 0 and out.num_observables == 1


def test_generated_family_matches_explicit_circuit():
    gen = decoder_compare(
        DecoderCompareInput(
            code_family="repetition_code", distance=3, p=0.01, decoders=["pymatching"], seed=3
        )
    )
    assert gen.ok and gen.num_detectors > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither source
        {"circuit": "H 0", "code_family": "color_code"},  # both sources
        {"code_family": "color_code", "distance": 4},  # even distance
    ],
)
def test_input_validation(kwargs):
    with pytest.raises(ValidationError):
        DecoderCompareInput(**kwargs)


def test_no_observables_is_reported():
    out = decoder_compare(
        DecoderCompareInput(circuit="R 0\nM 0\nDETECTOR rec[-1]", decoders=["pymatching"])
    )
    assert out.ok is False and "observable" in out.error_message


def test_missing_extra_fails_per_decoder_not_whole_call(monkeypatch):
    def no_bloqade(*a, **k):
        raise ImportError("no bloqade")

    monkeypatch.setattr("chatqec_mcp_tools.tools.decoder_compare.predict_isolated", no_bloqade)
    out = decoder_compare(
        DecoderCompareInput(circuit=_circuit(), decoders=["pymatching", "bposd"], shots=200)
    )
    by_name = {r.decoder: r for r in out.results}
    assert out.ok and out.best == "pymatching"
    assert by_name["bposd"].ok is False and "[decoders]" in by_name["bposd"].note


def test_noiseless_circuit_is_rejected_before_decoding():
    noiseless = str(stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3))
    out = decoder_compare(DecoderCompareInput(circuit=noiseless, decoders=["bposd"]))
    assert out.ok is False and "no noise" in out.error_message


@needs_decoders
def test_native_decoder_crash_is_isolated():
    """ldpc BP+OSD segfaults on this star-shaped DEM; the server process must survive."""
    import numpy as np

    from chatqec_mcp_tools.tools._isolated_decode import predict_isolated

    dem = "\n".join(["error(0.01) D0 L0"] + [f"error(0.01) D0 D{i}" for i in range(1, 6)])
    with pytest.raises(RuntimeError, match="crashed in native code"):
        predict_isolated("bposd", dem, np.zeros((2, 6), dtype=bool), timeout=60)


@needs_decoders
@pytest.mark.parametrize("family,d", [("bb_code", 6), ("toric_code", 3)])
def test_lightstim_families_are_built_with_noise(family, d):
    pytest.importorskip("lightstim")
    out = decoder_compare(
        DecoderCompareInput(
            code_family=family, distance=d, rounds=2, p=0.002, decoders=["bplsd"], shots=200, seed=1
        )
    )
    assert out.ok, out.error_message
    assert out.num_observables >= 1 and out.results[0].ok


@needs_decoders
def test_tesseract_beats_mwpm_on_color_code():
    out = decoder_compare(
        DecoderCompareInput(
            code_family="color_code",
            distance=3,
            p=0.004,
            decoders=["pymatching", "tesseract"],
            shots=3000,
            seed=7,
        )
    )
    by_name = {r.decoder: r for r in out.results}
    assert by_name["tesseract"].ok and by_name["pymatching"].ok
    assert by_name["tesseract"].logical_error_rate < by_name["pymatching"].logical_error_rate
    assert out.best == "tesseract"


@needs_decoders
def test_tesseract_shots_are_capped():
    out = decoder_compare(
        DecoderCompareInput(
            code_family="repetition_code", distance=3, p=0.01, decoders=["tesseract"], shots=6000
        )
    )
    (r,) = out.results
    assert r.shots == 5000 and "capped" in r.note


@needs_decoders
@pytest.mark.parametrize("name", ["bposd", "bplsd", "belief_find"])
def test_bp_family_decoders_run(name):
    out = decoder_compare(
        DecoderCompareInput(circuit=_circuit(), decoders=[name], shots=300, seed=2)
    )
    assert out.ok and out.results[0].logical_error_rate < 0.2


def _cudaq_gpu_available() -> bool:
    try:
        import cudaq
        import cudaq_qec  # noqa: F401
    except ImportError:
        return False
    return cudaq.num_available_gpus() > 0


needs_cudaq_gpu = pytest.mark.skipif(
    not _cudaq_gpu_available(), reason="cudaq extra or NVIDIA GPU not available"
)


def test_missing_cudaq_extra_points_at_cudaq_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("cudaq"):
            raise ImportError("no cudaq")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = decoder_compare(
        DecoderCompareInput(
            circuit=_circuit(), decoders=["pymatching", "cudaq_relay_bp"], shots=100
        )
    )
    by_name = {r.decoder: r for r in out.results}
    assert out.ok and by_name["cudaq_relay_bp"].ok is False
    assert "[cudaq]" in by_name["cudaq_relay_bp"].note


@needs_cudaq_gpu
@pytest.mark.parametrize("name", ["cudaq_relay_bp", "cudaq_bposd"])
def test_cudaq_gpu_decoders_beat_mwpm_on_color_code(name):
    out = decoder_compare(
        DecoderCompareInput(
            code_family="color_code",
            distance=3,
            p=0.004,
            decoders=["pymatching", name],
            shots=3000,
            seed=7,
        )
    )
    by_name = {r.decoder: r for r in out.results}
    assert by_name[name].ok, by_name[name].note
    assert by_name[name].logical_error_rate < by_name["pymatching"].logical_error_rate
