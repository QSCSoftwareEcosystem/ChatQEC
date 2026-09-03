from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest
from pydantic import ValidationError


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def test_ibm_submit_execution_rejects_missing_backend():
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(circuit_preset="bell", shots=100)  # backend omitted


def test_ibm_submit_execution_rejects_too_many_shots(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    out = ibm_submit_execution(
        IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=5000)
    )

    assert out.ok is False
    assert "shots" in out.error_message.lower()


def test_ibm_submit_execution_rejects_too_many_qubits(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    out = ibm_submit_execution(
        IbmSubmitExecutionInput(circuit_preset="ghz", num_qubits=11, backend="ibm_fake", shots=100)
    )

    assert out.ok is False
    assert "qubit" in out.error_message.lower()


def test_ibm_submit_execution_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    fake_job = SimpleNamespace(job_id=lambda: "job-123", status=lambda: "QUEUED")
    fake_sampler = SimpleNamespace(run=lambda *a, **kw: fake_job)
    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    # compute_mapping lives in map_circuit.py and calls get_ibm_service() using
    # THAT module's imported name -- patch it there, not in submit_execution's
    # namespace (this tool has no get_ibm_service call of its own).
    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
        patch("chatqec_mcp_tools.tools.ibm_hardware.submit_execution._make_sampler", return_value=fake_sampler),
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    assert out.job_id == "job-123"
    assert out.status == "QUEUED"


def test_ibm_submit_execution_qaoa_layers_out_of_bounds_rejected():
    """Part A: qaoa_layers must be bounded the same way Task 12 bounded it on
    IbmPreviewExecutionInput -- unbounded qaoa_layers is a cheap-to-request,
    expensive-to-build hazard (compute_mapping's construction is O(n * layers))
    independent of the credit-spend caps this tool otherwise enforces at
    runtime."""
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(backend="ibm_fake", shots=100, qaoa_layers=0)

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(backend="ibm_fake", shots=100, qaoa_layers=1001)


def test_ibm_submit_execution_handles_sampler_run_raising(monkeypatch, tmp_path):
    """sampler.run is a real network operation; if it raises, ibm_submit_execution
    must return ok=False rather than let the exception escape."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    def _boom(*a, **kw):
        raise RuntimeError("network exploded")

    fake_sampler = SimpleNamespace(run=_boom)
    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
        patch("chatqec_mcp_tools.tools.ibm_hardware.submit_execution._make_sampler", return_value=fake_sampler),
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert "network exploded" in out.error_message


def test_ibm_submit_execution_handles_job_id_raising(monkeypatch, tmp_path):
    """job.job_id()/job.status() are also real (potentially networked) calls on
    the returned job object; a failure there must not escape either."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    def _boom():
        raise RuntimeError("job_id lookup failed")

    fake_job = SimpleNamespace(job_id=_boom, status=lambda: "QUEUED")
    fake_sampler = SimpleNamespace(run=lambda *a, **kw: fake_job)
    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
        patch("chatqec_mcp_tools.tools.ibm_hardware.submit_execution._make_sampler", return_value=fake_sampler),
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert "job_id lookup failed" in out.error_message
