def test_tsim_simulate_registered():
    from chatqec_mcp_tools.server import build_server

    assert "tsim_simulate" in {t.name for t in build_server().list_tools()}


def test_missing_tsim_returns_ok_false(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "tsim":
            raise ImportError("no tsim")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from chatqec_mcp_tools.tools.tsim_simulate import (
        TsimSimulateInput,
        tsim_simulate,
    )

    out = tsim_simulate(TsimSimulateInput(circuit="H 0\nM 0", shots=10))
    assert out.ok is False
    assert "tsim" in out.error_message.lower()


def test_t_gate_circuit_with_real_tsim():
    """Bonus integration check: skipped when the optional `tsim` extra isn't
    installed; exercises a non-Clifford circuit Stim itself cannot run."""
    import pytest

    pytest.importorskip("tsim")

    from chatqec_mcp_tools.tools.tsim_simulate import (
        TsimSimulateInput,
        tsim_simulate,
    )

    out = tsim_simulate(
        TsimSimulateInput(
            circuit="RX 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]",
            shots=200,
        )
    )
    assert out.ok is True
    assert out.shots == 200
