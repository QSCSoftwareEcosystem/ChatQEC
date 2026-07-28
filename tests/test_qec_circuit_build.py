def test_qec_circuit_build_registered():
    from chatqec_mcp_tools.server import build_server

    assert "qec_circuit_build" in {t.name for t in build_server().list_tools()}


def test_missing_lightstim_returns_ok_false(monkeypatch):
    import chatqec_mcp_tools.lightstim_adapter as mod

    monkeypatch.setattr(
        mod,
        "build_circuit",
        lambda **_: (_ for _ in ()).throw(mod.LightStimUnavailable("lightstim not installed")),
    )
    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="toric_code", distance=3, rounds=3)
    )
    assert out.ok is False
    assert "lightstim" in out.error_message.lower()


def test_toric_code_build_with_real_lightstim():
    """Bonus integration check: skipped when the optional `lightstim` extra isn't
    installed; exercises the real adapter end-to-end (not a mocked build_circuit)."""
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="toric_code", distance=3, rounds=3)
    )
    assert out.ok is True
    assert out.num_qubits > 0
    assert out.num_detectors > 0
    assert out.num_observables == 2  # toric code encodes 2 logical qubits
