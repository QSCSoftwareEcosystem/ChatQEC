"""deltakit_memory: Riverlane Deltakit memory experiments with the local PyMatching decoder."""

import builtins

import pytest
from pydantic import ValidationError

from chatqec_mcp_tools.tools.deltakit_memory import DeltakitMemoryInput, deltakit_memory

pytest.importorskip("deltakit")


def _run(**kwargs):
    kwargs.setdefault("seed", 1)
    return deltakit_memory(DeltakitMemoryInput(**kwargs))


def test_rotated_surface_memory_d3_runs_locally():
    out = _run(shots=3000)
    assert out.ok, out.error_message
    assert out.decoder == "pymatching" and out.shots == 3000
    assert 0 <= out.logical_error_rate < 0.2
    assert out.num_qubits == 17 and out.num_detectors == 24 and out.num_observables == 1


def test_same_seed_gives_the_same_result():
    a = _run(p=0.01, shots=3000)
    b = _run(p=0.01, shots=3000)
    assert a.num_failures == b.num_failures


def test_higher_noise_gives_a_higher_error_rate():
    low = _run(p=0.001, shots=5000)
    high = _run(p=0.02, shots=5000)
    assert high.logical_error_rate > low.logical_error_rate


def test_larger_distance_suppresses_errors_below_threshold():
    d3 = _run(distance=3, p=0.002, shots=20000)
    d5 = _run(distance=5, p=0.002, shots=20000)
    assert d5.logical_error_rate < d3.logical_error_rate


@pytest.mark.parametrize(
    "family,observables",
    [("unrotated_planar", 1), ("unrotated_toric", 2), ("repetition", 1)],
)
def test_other_code_families_build_and_decode(family, observables):
    out = _run(code_family=family, shots=500)
    assert out.ok, out.error_message
    assert out.num_observables == observables


def test_x_basis_memory_runs():
    assert _run(logical_basis="X", shots=500).ok


def test_cloud_only_decoder_is_reported_as_not_configured():
    out = _run(decoder="lcd")
    assert out.ok is False
    assert "cloud" in out.error_message and "not configured" in out.error_message


def test_leakage_needs_the_cloud_and_says_so():
    out = _run(leakage_p=0.001)
    assert out.ok is False and "leakage" in out.error_message and "cloud" in out.error_message


def test_shots_are_capped():
    with pytest.raises(ValidationError):
        DeltakitMemoryInput(shots=100_000)


def test_tool_output_stays_off_the_mcp_stdout_channel(capfd):
    _run(shots=200)
    assert capfd.readouterr().out == ""


def test_missing_extra_points_at_the_deltakit_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.split(".")[0] in ("deltakit", "deltakit_circuit", "deltakit_core"):
            raise ImportError("missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = _run()
    assert out.ok is False and "[deltakit]" in out.error_message


def test_local_run_never_touches_the_network(monkeypatch):
    """The cloud costs money; the local pipeline must work with every connect forbidden."""
    import socket

    def no_network(*a, **k):
        raise AssertionError("the local Deltakit run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    out = _run(shots=300)
    assert out.ok, out.error_message
