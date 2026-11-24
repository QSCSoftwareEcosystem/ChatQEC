# tests/test_ibm_calibration.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _fake_service_with_graph_payload():
    # calibration.py serializes HardwareGraph to a plain dict before caching;
    # simulate that dict shape directly so this test doesn't depend on Task 4's
    # internals -- it patches build_hardware_graph, not the fake backend.
    class _Graph:
        def __init__(self):
            import networkx as nx

            g = nx.Graph()
            g.add_node(
                0,
                T1=3.0e-05,
                T2=2.7e-05,
                readout_error=0.0127,
                readout_fidelity=0.9873,
                is_operational=True,
            )
            g.add_node(
                1,
                T1=2.5e-05,
                T2=2.0e-05,
                readout_error=0.02,
                readout_fidelity=0.98,
                is_operational=True,
            )
            g.add_edge(
                0, 1, native_gate="cz", gate_error=0.0097, gate_time=6.8e-08, gate_fidelity=0.9903
            )
            self.graph = g
            self.backend_name = "ibm_fake"

    service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))
    return service, _Graph()


def test_ibm_backend_calibration_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    service, hw_graph = _fake_service_with_graph_payload()

    with (
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service", return_value=service
        ),
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.calibration.build_hardware_graph",
            return_value=hw_graph,
        ),
    ):
        out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is True
    assert out.backend == "ibm_fake"
    assert out.age_seconds == 0
    assert out.stale is False
    assert {q.index for q in out.qubits} == {0, 1}
    q0 = next(q for q in out.qubits if q.index == 0)
    assert q0.t1_us == pytest_approx(30.0)  # seconds -> microseconds
    assert len(out.couplers) == 1
    assert out.couplers[0].native_gate == "cz"


def test_ibm_backend_calibration_second_call_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    service, hw_graph = _fake_service_with_graph_payload()
    call_count = {"n": 0}

    def build_and_count(backend):
        call_count["n"] += 1
        return hw_graph

    with (
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service", return_value=service
        ),
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.calibration.build_hardware_graph",
            side_effect=build_and_count,
        ),
    ):
        ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))
        out2 = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert call_count["n"] == 1
    assert out2.age_seconds >= 0


def test_ibm_backend_calibration_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    with patch(
        "chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service",
        side_effect=IBMCredentialsError("no token configured"),
    ):
        out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is False
    assert "no token configured" in out.error_message


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


# --------------------------------------------------------------------------
# A cached payload whose SHAPE does not match must degrade, not raise.
# --------------------------------------------------------------------------


def _write_cache(tmp_path, key, payload):
    import json
    import time

    root = tmp_path / "chatqec-mcp-tools" / "ibm_hardware"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{key.replace('/', '__')}.json").write_text(
        json.dumps({"fetched_at": time.time(), "ttl_seconds": 1e9, "payload": payload}),
        encoding="utf-8",
    )


def test_ibm_backend_calibration_handles_missing_cached_keys(monkeypatch, tmp_path):
    """cache_result.payload["qubits"] / ["couplers"] were raw dict subscripts
    outside any try/except, so a cached payload lacking those keys raised
    KeyError straight out of the tool instead of returning ok=False."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    _write_cache(tmp_path, "calibration/ibm_fake", {"qubit_list": []})

    out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is False
    assert out.qubits == []
    assert out.couplers == []
    assert "unexpected format" in out.error_message
    assert "retry" in out.error_message.lower()


def test_ibm_backend_calibration_handles_cached_entry_shape_mismatch(monkeypatch, tmp_path):
    """Keys present but their contents no longer match QubitCalibration /
    CouplerCalibration -- pydantic's ValidationError must be caught too, not
    just KeyError."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    _write_cache(
        tmp_path,
        "calibration/ibm_fake",
        {"qubits": [{"qubit_index": 0, "t1_microseconds": 30.0}], "couplers": []},
    )

    out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is False
    assert "unexpected format" in out.error_message
