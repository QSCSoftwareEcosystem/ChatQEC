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


def test_color_code_666_build_with_real_lightstim():
    """color_code_666 happy path at a valid (odd) distance — real lightstim, no mocks."""
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="color_code_666", distance=3, rounds=3)
    )
    assert out.ok is True
    assert out.num_qubits == 13
    assert out.num_detectors == 18
    assert out.num_observables == 1  # color code encodes 1 logical qubit


def test_color_code_666_even_distance_returns_ok_false():
    """color_code_666 rejects even distance with a clear message — this is the
    adapter's own runtime validation (not pydantic-level), so it needs its own
    test regardless of whether lightstim is installed."""
    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="color_code_666", distance=4, rounds=2)
    )
    assert out.ok is False
    assert "odd" in out.error_message.lower()


def test_bb_code_build_with_real_lightstim():
    """bb_code happy path at a valid discrete preset distance (d=6) — real lightstim."""
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="bb_code", distance=6, rounds=3)
    )
    assert out.ok is True
    assert out.num_qubits == 144
    assert out.num_observables == 12


def test_bb_code_unsupported_distance_returns_ok_false():
    """bb_code only has discrete presets ({6, 10, 12} within this tool's supported
    distance range) — d=7 has no matching preset and must fail gracefully, not raise."""
    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="bb_code", distance=7, rounds=3)
    )
    assert out.ok is False
    assert "bb_code" in out.error_message
    assert "6" in out.error_message and "10" in out.error_message and "12" in out.error_message


def test_pqrm_build_with_real_lightstim():
    """pqrm happy path at a valid discrete preset distance (d=3) — real lightstim.

    Exercises the hand-rolled hypercube-encode circuit-building path (PQRM cannot
    go through the generic MemoryExperiment used by the other three families).
    """
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="pqrm", distance=3, rounds=3)
    )
    assert out.ok is True
    assert out.num_qubits == 25
    assert out.num_observables == 1


def test_pqrm_unsupported_distance_returns_ok_false():
    """pqrm only has logical-operator presets for distance in {3, 7} — d=5 has no
    matching (rx, rz, m) preset and must fail gracefully, not raise."""
    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="pqrm", distance=5, rounds=3)
    )
    assert out.ok is False
    assert "pqrm" in out.error_message
    assert "3" in out.error_message and "7" in out.error_message


def test_lightstim_api_error_returns_ok_false(monkeypatch):
    """Simulates upstream LightStim renaming/removing something so an
    ImportError/TypeError/AttributeError escapes the submodule-import layer
    inside the builder functions (executed *after* the top-level `import
    lightstim` availability gate passes). These must still surface as
    ok=False, not raise, per the tool's own advertised contract -- and with a
    message distinguishable from ordinary domain-validation ValueErrors."""
    import chatqec_mcp_tools.lightstim_adapter as mod

    monkeypatch.setattr(
        mod,
        "build_circuit",
        lambda **_: (_ for _ in ()).throw(
            AttributeError("module 'lightstim.ir.qec_system' has no attribute 'QECSystem'")
        ),
    )
    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="toric_code", distance=3, rounds=3)
    )
    assert out.ok is False
    assert "lightstim api error" in out.error_message.lower()


def test_work_ceiling_rejects_toric_code_worst_case_with_real_lightstim():
    """The reported worst case (toric_code d=15, r=200) measured ~3.6 minutes
    to build and, since qec_circuit_build's handler runs synchronously inside
    the MCP server's async call_tool, blocked the whole server for that
    duration. Must now be rejected up front (near-instantly) instead of
    actually attempted."""
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="toric_code", distance=15, rounds=200)
    )
    assert out.ok is False
    assert "work ceiling" in out.error_message.lower()


def test_circuit_text_size_cap_with_real_lightstim():
    """toric_code at distance=11 builds quickly (well under the work ceiling)
    but its circuit_text (~76 KB) exceeds stim_diagram's 64 KB input cap --
    must be rejected here too, so a build-then-diagram follow-up call doesn't
    silently receive an unusable circuit_text."""
    import pytest

    pytest.importorskip("lightstim")

    from chatqec_mcp_tools.tools.qec_circuit_build import (
        QecCircuitBuildInput,
        qec_circuit_build,
    )

    out = qec_circuit_build(
        QecCircuitBuildInput(code_family="toric_code", distance=11, rounds=3)
    )
    assert out.ok is False
    assert "too large" in out.error_message.lower()
