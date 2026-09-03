from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def test_ibm_map_circuit_success_with_explicit_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
    ):
        out = ibm_map_circuit(
            IbmMapCircuitInput(circuit_preset="bell", backend="ibm_fake", objective="maximum_fidelity")
        )

    assert out.ok is True
    assert out.backend == "ibm_fake"
    assert len(out.physical_qubits) == 2
    assert out.strategy_used in ("vf2", "greedy")
    assert 0.0 <= out.estimated_fidelity <= 1.0
    assert out.coupling_map_svg != ""


def test_ibm_map_circuit_auto_selects_least_busy_operational_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    busy = SimpleNamespace(
        name="ibm_busy", num_qubits=5, status=lambda: SimpleNamespace(operational=True, pending_jobs=9)
    )
    quiet = SimpleNamespace(
        name="ibm_quiet", num_qubits=5, status=lambda: SimpleNamespace(operational=True, pending_jobs=0)
    )
    down = SimpleNamespace(
        name="ibm_down", num_qubits=5, status=lambda: SimpleNamespace(operational=False, pending_jobs=0)
    )
    fake_service = SimpleNamespace(backends=lambda: [busy, down, quiet])

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
    ):
        out = ibm_map_circuit(IbmMapCircuitInput(circuit_preset="bell"))  # no backend given

    assert out.ok is True
    assert out.backend == "ibm_quiet"


def test_ibm_map_circuit_rejects_both_qasm_and_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    out = ibm_map_circuit(
        IbmMapCircuitInput(circuit_qasm="OPENQASM 2.0;", circuit_preset="bell", backend="ibm_fake")
    )

    assert out.ok is False
    assert "exactly one" in out.error_message


def test_ibm_map_circuit_rejects_circuit_too_big_for_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph(n=2)),
    ):
        out = ibm_map_circuit(IbmMapCircuitInput(circuit_preset="ghz", num_qubits=5, backend="ibm_fake"))

    assert out.ok is False
    assert "qubits" in out.error_message.lower()
