"""IQM tools with the worker mocked out (no network, no IQM env needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("qiskit")

_GRAPH = "chatqec_mcp_tools.tools.iqm_hardware._hardware_graph.call_worker"
_SUBMIT = "chatqec_mcp_tools.tools.iqm_hardware.submit_execution.call_worker"
_JOB = "chatqec_mcp_tools.tools.iqm_hardware.job_result.call_worker"


def _snapshot() -> dict:
    """A 2x3 grid: QB1-QB2-QB3 / QB4-QB5-QB6 with vertical links; QB5-QB6 is the best CZ."""
    qubits = [
        {
            "name": f"QB{i}",
            "t1_s": 50e-6,
            "t2_s": 10e-6,
            "t2_echo_s": 30e-6,
            "readout_fidelity": 0.97,
            "readout_error_0_to_1": 0.02,
            "readout_error_1_to_0": 0.04,
            "prx_fidelity": 0.999,
        }
        for i in range(1, 7)
    ]
    pairs = [
        ("QB1", "QB2"),
        ("QB2", "QB3"),
        ("QB4", "QB5"),
        ("QB5", "QB6"),
        ("QB1", "QB4"),
        ("QB2", "QB5"),
        ("QB3", "QB6"),
    ]
    couplers = [
        {
            "qubits": list(p),
            "cz_fidelity": 0.999 if p == ("QB5", "QB6") else 0.97,
            "clifford_cz_fidelity": 0.96,
            "cz_duration_s": 60e-9,
        }
        for p in pairs
    ]
    return {
        "quantum_computer": "default",
        "dut_label": "TEST_CHIP",
        "healthy": True,
        "calibration_set_id": "00000000-0000-0000-0000-000000000000",
        "native_gates": ["cz", "measure", "prx"],
        "qubits": qubits,
        "couplers": couplers,
    }


@pytest.fixture
def iqm_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("IQM_TOKEN", "test-token")
    monkeypatch.setenv("IQM_SERVER_URL", "https://iqm.example/")


def test_list_backends(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.list_backends import (
        IqmListBackendsInput,
        iqm_list_backends,
    )

    with patch(_GRAPH, return_value=_snapshot()):
        out = iqm_list_backends(IqmListBackendsInput())
    assert out.ok, out.error_message
    assert out.backends[0].num_qubits == 6
    assert out.backends[0].num_couplers == 7
    assert out.backends[0].healthy is True


def test_snapshot_is_cached(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware._hardware_graph import get_snapshot

    with patch(_GRAPH, return_value=_snapshot()) as worker:
        get_snapshot()
        second = get_snapshot()
    assert worker.call_count == 1
    assert second.age_seconds >= 0


def test_credentials_error_not_served_stale(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError
    from chatqec_mcp_tools.tools.iqm_hardware.list_backends import (
        IqmListBackendsInput,
        iqm_list_backends,
    )

    with patch(_GRAPH, side_effect=IQMCredentialsError("bad token")):
        out = iqm_list_backends(IqmListBackendsInput())
    assert not out.ok
    assert "bad token" in out.error_message


def test_calibration_units(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.calibration import (
        IqmBackendCalibrationInput,
        iqm_backend_calibration,
    )

    with patch(_GRAPH, return_value=_snapshot()):
        out = iqm_backend_calibration(IqmBackendCalibrationInput())
    assert out.ok
    assert out.qubits[0].t1_us == pytest.approx(50.0)
    assert out.qubits[0].t2_echo_us == pytest.approx(30.0)
    assert out.couplers[0].cz_duration_ns == pytest.approx(60.0)


def test_graph_matches_ibm_schema(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware._hardware_graph import build_hardware_graph

    hw = build_hardware_graph(_snapshot())
    assert hw.graph.nodes[0]["name"] == "QB1"
    assert hw.graph.nodes[0]["readout_error"] == pytest.approx(0.03)
    assert hw.graph.nodes[0]["T2"] == pytest.approx(30e-6)  # echo T2, like IBM's
    edge = hw.graph.edges[4, 5]  # QB5-QB6
    assert edge["gate_fidelity"] == pytest.approx(0.999)
    assert edge["gate_error"] == pytest.approx(0.001)


def test_map_bell_picks_best_coupler(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import (
        IqmMapCircuitInput,
        iqm_map_circuit,
    )

    with patch(_GRAPH, return_value=_snapshot()):
        out = iqm_map_circuit(IqmMapCircuitInput(circuit_preset="bell"))
    assert out.ok, out.error_message
    assert set(out.physical_qubits) == {"QB5", "QB6"}
    assert out.estimated_fidelity == pytest.approx(0.999 * 0.97 * 0.97)


def test_map_too_many_qubits(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import (
        IqmMapCircuitInput,
        iqm_map_circuit,
    )

    with patch(_GRAPH, return_value=_snapshot()):
        out = iqm_map_circuit(IqmMapCircuitInput(circuit_preset="ghz", num_qubits=7))
    assert not out.ok
    assert "only 6" in out.error_message


def test_preview_does_not_submit(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.preview_execution import (
        IqmPreviewExecutionInput,
        iqm_preview_execution,
    )

    with patch(_GRAPH, return_value=_snapshot()), patch(_SUBMIT) as submit:
        out = iqm_preview_execution(IqmPreviewExecutionInput(circuit_preset="bell", shots=100))
    assert out.ok
    assert out.backend_healthy is True
    submit.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"circuit_preset": "bell", "shots": 100}, "confirm_submit"),
        ({"circuit_preset": "bell", "shots": 5000, "confirm_submit": True}, "cap of 4000"),
        (
            {"circuit_preset": "ghz", "num_qubits": 11, "shots": 10, "confirm_submit": True},
            "cap of 10",
        ),
    ],
)
def test_submit_refusals_never_call_worker(iqm_env, kwargs, message):
    from chatqec_mcp_tools.tools.iqm_hardware.submit_execution import (
        IqmSubmitExecutionInput,
        iqm_submit_execution,
    )

    with patch(_GRAPH, return_value=_snapshot()), patch(_SUBMIT) as submit:
        out = iqm_submit_execution(IqmSubmitExecutionInput(**kwargs))
    assert not out.ok
    assert message in out.error_message
    submit.assert_not_called()


def test_submit_refuses_unhealthy_device(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.submit_execution import (
        IqmSubmitExecutionInput,
        iqm_submit_execution,
    )

    snap = {**_snapshot(), "healthy": False}
    with patch(_GRAPH, return_value=snap), patch(_SUBMIT) as submit:
        out = iqm_submit_execution(
            IqmSubmitExecutionInput(circuit_preset="bell", shots=10, confirm_submit=True)
        )
    assert not out.ok
    submit.assert_not_called()


def test_submit_sends_measured_qasm_and_layout(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.submit_execution import (
        IqmSubmitExecutionInput,
        iqm_submit_execution,
    )

    reply = {
        "job_id": "abc-123",
        "status": "QUEUED",
        "transpiled_two_qubit_gate_count": 1,
        "transpiled_depth": 4,
    }
    with patch(_GRAPH, return_value=_snapshot()), patch(_SUBMIT, return_value=reply) as submit:
        out = iqm_submit_execution(
            IqmSubmitExecutionInput(circuit_preset="bell", shots=100, confirm_submit=True)
        )
    assert out.ok, out.error_message
    assert out.job_id == "abc-123"
    args = submit.call_args
    assert args.args == ("submit",)
    assert set(args.kwargs["initial_layout"]) == {"QB5", "QB6"}
    assert "measure" in args.kwargs["qasm"]
    assert args.kwargs["shots"] == 100


def test_job_result(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.job_result import IqmJobResultInput, iqm_job_result

    with patch(_JOB, return_value={"status": "DONE", "counts": {"00": 48, "11": 52}}):
        out = iqm_job_result(IqmJobResultInput(job_id="0f1e-22"))
    assert out.ok
    assert out.counts == {"00": 48, "11": 52}


def test_job_result_rejects_odd_id():
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.iqm_hardware.job_result import IqmJobResultInput

    with pytest.raises(ValidationError):
        IqmJobResultInput(job_id="../../etc")


# --- the subprocess boundary -------------------------------------------------------


def _fake_worker(tmp_path: Path, reply: dict) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, os, sys\n"
        "req = json.loads(sys.stdin.read())\n"
        f"reply = {reply!r}\n"
        "reply.setdefault('result', {})\n"
        "reply['result']['env_keys'] = sorted(os.environ)\n"
        "reply['result']['op'] = req['op']\n"
        "print(json.dumps(reply))\n"
    )
    return script


def test_call_worker_passes_only_iqm_env(iqm_env, monkeypatch, tmp_path):
    from chatqec_mcp_tools.tools.iqm_hardware import _client

    monkeypatch.setenv("CHATQEC_IQM_PYTHON", sys.executable)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setattr(_client, "_WORKER", _fake_worker(tmp_path, {"ok": True}))
    result = _client.call_worker("snapshot")
    assert result["op"] == "snapshot"
    assert "IQM_TOKEN" in result["env_keys"]
    assert "ANTHROPIC_API_KEY" not in result["env_keys"]


def test_call_worker_auth_failure(iqm_env, monkeypatch, tmp_path):
    from chatqec_mcp_tools.tools.iqm_hardware import _client

    monkeypatch.setenv("CHATQEC_IQM_PYTHON", sys.executable)
    monkeypatch.setattr(
        _client, "_WORKER", _fake_worker(tmp_path, {"ok": False, "kind": "auth", "error": "401"})
    )
    with pytest.raises(_client.IQMCredentialsError):
        _client.call_worker("snapshot")


def test_call_worker_missing_settings(monkeypatch):
    from chatqec_mcp_tools.tools.iqm_hardware import _client

    monkeypatch.delenv("IQM_TOKEN", raising=False)
    with pytest.raises(_client.IQMCredentialsError, match="IQM_TOKEN"):
        _client.call_worker("snapshot")
    monkeypatch.setenv("IQM_TOKEN", "t")
    monkeypatch.delenv("IQM_SERVER_URL", raising=False)
    with pytest.raises(_client.IQMCredentialsError, match="IQM_SERVER_URL"):
        _client.call_worker("snapshot")


def test_call_worker_missing_env(iqm_env, monkeypatch, tmp_path):
    from chatqec_mcp_tools.tools.iqm_hardware import _client

    monkeypatch.setenv("CHATQEC_IQM_PYTHON", str(tmp_path / "nope" / "python"))
    with pytest.raises(_client.IQMCredentialsError, match="setup_iqm_env"):
        _client.call_worker("snapshot")


def test_worker_redacts_token(monkeypatch, capsys):
    """The worker never echoes the token, even inside an exception message."""
    from chatqec_mcp_tools.tools.iqm_hardware import _worker

    monkeypatch.setenv("IQM_TOKEN", "sekrit-token-value")
    monkeypatch.setenv("IQM_SERVER_URL", "https://iqm.example/")

    def boom(_req):
        raise RuntimeError("server said no for sekrit-token-value")

    monkeypatch.setitem(_worker._OPS, "snapshot", boom)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"op": "snapshot"})))
    _worker.main()
    reply = json.loads(capsys.readouterr().out)
    assert not reply["ok"]
    assert "sekrit-token-value" not in reply["error"]
