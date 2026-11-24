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


def test_ibm_preview_execution_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
        IbmPreviewExecutionInput,
        ibm_preview_execution,
    )

    # compute_mapping lives in map_circuit.py and calls get_ibm_service() using
    # THAT module's imported name -- patch it there, not in preview_execution's
    # namespace, or the real (uncredentialed) SDK call runs during the test.
    fake_service = SimpleNamespace(
        backend=lambda name: SimpleNamespace(
            name=name, status=lambda: SimpleNamespace(operational=True, pending_jobs=2)
        )
    )

    with (
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service",
            return_value=fake_service,
        ),
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph",
            return_value=_fake_hw_graph(),
        ),
    ):
        out = ibm_preview_execution(
            IbmPreviewExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=500)
        )

    assert out.ok is True
    assert out.shots == 500
    assert out.estimated_queue_position == 2
    assert out.backend_operational is True
    assert len(out.physical_qubits) == 2


def test_ibm_preview_execution_propagates_mapping_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
        IbmPreviewExecutionInput,
        ibm_preview_execution,
    )

    out = ibm_preview_execution(IbmPreviewExecutionInput(backend="ibm_fake"))  # no circuit given

    assert out.ok is False
    assert "exactly one" in out.error_message


def test_ibm_preview_execution_num_qubits_out_of_bounds_rejected():
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import IbmPreviewExecutionInput

    with pytest.raises(ValidationError):
        IbmPreviewExecutionInput(backend="ibm_fake", num_qubits=0)

    with pytest.raises(ValidationError):
        IbmPreviewExecutionInput(backend="ibm_fake", num_qubits=1025)


def test_ibm_preview_execution_qaoa_layers_out_of_bounds_rejected():
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import IbmPreviewExecutionInput

    with pytest.raises(ValidationError):
        IbmPreviewExecutionInput(backend="ibm_fake", qaoa_layers=0)

    with pytest.raises(ValidationError):
        IbmPreviewExecutionInput(backend="ibm_fake", qaoa_layers=1001)
