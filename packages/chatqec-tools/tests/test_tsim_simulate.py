import pytest


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


def test_circuit_too_large_returns_ok_false():
    """Extra-independent: the size cap is checked before `import tsim`, so
    this must reject even when the `tsim` extra isn't installed."""
    from chatqec_mcp_tools.tools.tsim_simulate import (
        _MAX_CIRCUIT_BYTES,
        TsimSimulateInput,
        tsim_simulate,
    )

    oversized = "H 0\n" * (_MAX_CIRCUIT_BYTES // len(b"H 0\n") + 1)
    out = tsim_simulate(TsimSimulateInput(circuit=oversized, shots=1))
    assert out.ok is False
    assert "too large" in out.error_message.lower()


def test_too_many_shots_returns_ok_false():
    """Extra-independent: the shots cap is checked before `import tsim`, so
    this must reject even when the `tsim` extra isn't installed."""
    from chatqec_mcp_tools.tools.tsim_simulate import (
        _MAX_SHOTS,
        TsimSimulateInput,
        tsim_simulate,
    )

    out = tsim_simulate(
        TsimSimulateInput(circuit="H 0\nM 0\nDETECTOR rec[-1]", shots=_MAX_SHOTS + 1)
    )
    assert out.ok is False
    assert "shots" in out.error_message.lower()


def test_malformed_detector_rec_returns_ok_false_with_real_tsim():
    """Regression test for the escaped IndexError: a DETECTOR referencing a
    measurement record that doesn't exist must return ok=False, not raise."""
    import pytest

    pytest.importorskip("tsim")

    from chatqec_mcp_tools.tools.tsim_simulate import (
        TsimSimulateInput,
        tsim_simulate,
    )

    out = tsim_simulate(TsimSimulateInput(circuit="DETECTOR rec[-1]", shots=1))
    assert out.ok is False
    assert "error" in out.error_message.lower()


def test_no_detector_or_observable_returns_ok_false_with_real_tsim():
    """An M-only circuit has nothing for compile_detector_sampler() to sample
    (it returns shape (shots, 0)) — must be surfaced as ok=False rather than
    a misleading ok=True with an empty preview."""
    import pytest

    pytest.importorskip("tsim")

    from chatqec_mcp_tools.tools.tsim_simulate import (
        TsimSimulateInput,
        tsim_simulate,
    )

    out = tsim_simulate(TsimSimulateInput(circuit="H 0\nM 0", shots=4))
    assert out.ok is False
    assert "detector" in out.error_message.lower()


def test_work_ceiling_rejects_high_tcount_circuit_with_real_tsim():
    """Reported worst case: a schema-valid, 233-byte T-gate circuit measured
    taking >17 minutes (killed unfinished), blocking the async server's event
    loop. A circuit whose non-Clifford tcount is over the ceiling must now be
    rejected near-instantly instead of actually attempted."""
    import time

    import pytest

    pytest.importorskip("tsim")

    from chatqec_mcp_tools.tools.tsim_simulate import (
        _MAX_TCOUNT,
        TsimSimulateInput,
        tsim_simulate,
    )

    lines = ["RX 0", "RX 1", "RX 2", "RX 3"]
    for i in range(_MAX_TCOUNT + 10):
        q = i % 4
        lines += [f"T {q}", f"H {q}", f"CX {q} {(q + 1) % 4}"]
    lines += ["M 0", "M 1", "M 2", "M 3", "DETECTOR rec[-1]"]
    circuit_text = "\n".join(lines)

    start = time.monotonic()
    out = tsim_simulate(TsimSimulateInput(circuit=circuit_text, shots=10))
    elapsed = time.monotonic() - start

    assert out.ok is False
    assert "tcount" in out.error_message.lower()
    assert elapsed < 2.0


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


def test_column_rates_match_t_gate_physics():
    """T on |+>, measured in X: the detector fires with sin^2(pi/8); T then T_DAG never."""
    pytest.importorskip("tsim")
    import math

    from chatqec_mcp_tools.tools.tsim_simulate import TsimSimulateInput, tsim_simulate

    t = tsim_simulate(TsimSimulateInput(circuit="RX 0\nT 0\nMX 0\nDETECTOR rec[-1]", shots=20000))
    ident = tsim_simulate(
        TsimSimulateInput(circuit="RX 0\nT 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]", shots=2000)
    )
    assert t.ok and ident.ok
    assert abs(t.column_rates[0] - math.sin(math.pi / 8) ** 2) < 0.015
    assert ident.column_rates == [0.0]
