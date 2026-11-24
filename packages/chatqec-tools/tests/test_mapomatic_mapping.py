"""mapper="mapomatic" for the IBM and IQM mapping tools (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("mapomatic")

from test_iqm_hardware import _snapshot

_IQM_GRAPH = "chatqec_mcp_tools.tools.iqm_hardware._hardware_graph.call_worker"
_IQM_SUBMIT = "chatqec_mcp_tools.tools.iqm_hardware.submit_execution.call_worker"


@pytest.fixture
def iqm_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("IQM_TOKEN", "test-token")
    monkeypatch.setenv("IQM_SERVER_URL", "https://iqm.example/")


def _line_graph(n: int, bad_edge: tuple[int, int] | None = None) -> nx.Graph:
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, one_qubit_error=0.001, readout_error=0.02)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_error=0.2 if (q, q + 1) == bad_edge else 0.01)
    return g


def test_fidelity_cost_counts_every_gate():
    from qiskit import QuantumCircuit

    from chatqec_mcp_tools.tools.ibm_hardware._mapomatic import fidelity_cost

    qc = QuantumCircuit(2, 2)
    qc.sx(0)
    qc.rz(0.3, 1)  # virtual, free
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])
    [(_, err)] = fidelity_cost(qc, [[0, 1]], _line_graph(3))
    expected = 1 - (1 - 0.001) * (1 - 0.01) * (1 - 0.02) ** 2
    assert err == pytest.approx(expected)


def test_fidelity_cost_rejects_missing_coupler():
    from qiskit import QuantumCircuit

    from chatqec_mcp_tools.tools.ibm_hardware._mapomatic import fidelity_cost

    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    [(_, err)] = fidelity_cost(qc, [[0, 2]], _line_graph(3))
    assert err == pytest.approx(1.0)


def test_mapomatic_place_avoids_bad_coupler():
    from qiskit import QuantumCircuit

    from chatqec_mcp_tools.tools.ibm_hardware._mapomatic import iqm_target, mapomatic_place

    hw = _line_graph(4, bad_edge=(0, 1))
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    res = mapomatic_place(qc, hw, target=iqm_target(hw))
    assert (0, 1) not in {tuple(sorted(res.layout))}
    assert set(res.assignment) == {0, 1}
    assert res.num_candidate_layouts >= 3
    assert res.two_qubit_gate_count == 1
    assert 0.0 < res.estimated_fidelity < 1.0


def test_iqm_map_with_mapomatic(iqm_env):
    from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import (
        IqmMapCircuitInput,
        iqm_map_circuit,
    )

    with patch(_IQM_GRAPH, return_value=_snapshot()):
        out = iqm_map_circuit(IqmMapCircuitInput(circuit_preset="bell", mapper="mapomatic"))
    assert out.ok, out.error_message
    assert out.mapper == "mapomatic"
    assert out.strategy_used == "mapomatic"
    assert set(out.physical_qubits) == {"QB5", "QB6"}  # the best CZ in the fixture
    assert out.num_candidate_layouts > 1
    assert set(out.logical_to_physical) == {"0", "1"}


def test_iqm_submit_with_mapomatic_sends_routed_circuit(iqm_env):
    from qiskit import QuantumCircuit

    from chatqec_mcp_tools.tools.iqm_hardware.submit_execution import (
        IqmSubmitExecutionInput,
        iqm_submit_execution,
    )

    reply = {"job_id": "j-1", "status": "QUEUED"}
    with patch(_IQM_GRAPH, return_value=_snapshot()), patch(_IQM_SUBMIT, return_value=reply) as w:
        out = iqm_submit_execution(
            IqmSubmitExecutionInput(
                circuit_preset="ghz",
                num_qubits=3,
                shots=10,
                confirm_submit=True,
                mapper="mapomatic",
            )
        )
    assert out.ok, out.error_message
    kwargs = w.call_args.kwargs
    routed = QuantumCircuit.from_qasm_str(kwargs["qasm"])  # the worker must be able to parse it
    assert routed.num_qubits == len(kwargs["initial_layout"])
    assert set(routed.count_ops()) <= {"r", "cz", "measure", "barrier"}
    assert all(name.startswith("QB") for name in kwargs["initial_layout"])


def test_ibm_map_with_mapomatic():
    from qiskit.providers.fake_provider import GenericBackendV2

    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    backend = GenericBackendV2(
        5,
        basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
        seed=7,
    )
    hw = SimpleNamespace(graph=_line_graph(5, bad_edge=(0, 1)), backend_name="fake")
    service = SimpleNamespace(backend=lambda _name: backend)
    with (
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=service
        ),
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=hw
        ),
    ):
        out = ibm_map_circuit(
            IbmMapCircuitInput(circuit_preset="bell", backend="fake", mapper="mapomatic")
        )
    assert out.ok, out.error_message
    assert out.mapper == "mapomatic"
    assert sorted(out.physical_qubits) != [0, 1]
    assert out.transpiled_two_qubit_gate_count >= 1


def test_ibm_submit_with_mapomatic_runs_routed_layout(monkeypatch, tmp_path):
    from qiskit.providers.fake_provider import GenericBackendV2

    from chatqec_mcp_tools.tools.ibm_hardware import submit_execution as sub

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    backend = GenericBackendV2(
        4,
        basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        seed=7,
    )
    hw = SimpleNamespace(graph=_line_graph(4), backend_name="fake")
    service = SimpleNamespace(backend=lambda _name: backend)
    captured = {}

    def fake_transpile(circuit, _backend, initial_layout):
        captured["circuit"], captured["layout"] = circuit, initial_layout
        return circuit

    job = SimpleNamespace(job_id=lambda: "ibm-job", status=lambda: "QUEUED")
    sampler = SimpleNamespace(run=lambda *_a, **_k: job)
    with (
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=service
        ),
        patch(
            "chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=hw
        ),
        patch.object(sub, "_transpile", side_effect=fake_transpile),
        patch.object(sub, "_make_sampler", return_value=sampler),
    ):
        out = sub.ibm_submit_execution(
            sub.IbmSubmitExecutionInput(
                circuit_preset="bell", backend="fake", shots=10, mapper="mapomatic"
            )
        )
    assert out.ok, out.error_message
    assert out.job_id == "ibm-job"
    assert captured["circuit"].num_qubits == len(captured["layout"])
    assert captured["circuit"].count_ops().get("measure", 0) == 2
