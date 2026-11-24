from __future__ import annotations

from types import SimpleNamespace


class _FakeProperties:
    """Mimics qiskit's BackendProperties interface for the two calls
    build_hardware_graph needs: qubit_property(q) and gate_error/gate_length."""

    def __init__(self, qubits: dict[int, dict], gates: dict[tuple, dict]):
        self._qubits = qubits
        self._gates = gates  # {(gate_name, (u, v)): {"error": ..., "length": ...}}
        self.last_update_date = "2026-09-03T14:14:15-04:00"

    def qubit_property(self, q: int) -> dict:
        return self._qubits.get(q, {})

    def gate_error(self, gate: str, qubits) -> float:
        key = (gate, tuple(qubits))
        if key not in self._gates:
            raise ValueError(f"no calibration for {gate} on {qubits}")
        return self._gates[key]["error"]

    def gate_length(self, gate: str, qubits) -> float:
        key = (gate, tuple(qubits))
        return self._gates[key]["length"]


def _fake_backend():
    props = _FakeProperties(
        qubits={
            0: {"T1": (3.0e-05, "s"), "T2": (2.7e-05, "s"), "readout_error": (0.0127, "")},
            1: {"T1": (2.5e-05, "s"), "T2": (2.0e-05, "s"), "readout_error": (0.02, "")},
        },
        gates={("cz", (0, 1)): {"error": 0.0097, "length": 6.8e-08}},
    )
    return SimpleNamespace(
        name="ibm_fake",
        num_qubits=2,
        coupling_map=[(0, 1), (1, 0)],
        properties=lambda: props,
    )


def test_build_hardware_graph_populates_node_attrs():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    assert hg.backend_name == "ibm_fake"
    assert hg.num_qubits == 2
    q0 = hg.graph.nodes[0]
    assert q0["T1"] == 3.0e-05
    assert q0["readout_fidelity"] == pytest_approx(1 - 0.0127)
    assert q0["is_operational"] is True


def test_build_hardware_graph_picks_native_gate_by_priority():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    edge = hg.graph.edges[0, 1]
    assert edge["native_gate"] == "cz"
    assert edge["gate_error"] == 0.0097
    assert edge["gate_fidelity"] == pytest_approx(1 - 0.0097)


def test_build_hardware_graph_deduplicates_bidirectional_coupling_pairs():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    assert hg.graph.number_of_edges() == 1  # (0,1) and (1,0) collapse to one undirected edge


def test_is_operational_false_when_t1_missing():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    props = _FakeProperties(
        qubits={0: {"readout_error": (0.01, "")}, 1: {"T1": (2.5e-05, "s"), "T2": (2.0e-05, "s")}},
        gates={},
    )
    backend = SimpleNamespace(
        name="ibm_fake2", num_qubits=2, coupling_map=[(0, 1)], properties=lambda: props
    )

    hg = build_hardware_graph(backend)

    assert hg.graph.nodes[0]["is_operational"] is False


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
