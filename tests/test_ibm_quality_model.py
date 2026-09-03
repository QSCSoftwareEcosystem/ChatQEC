from __future__ import annotations

import networkx as nx


def _build_graph():
    g = nx.Graph()
    g.add_node(0, T1=30.0, T2=27.0, readout_fidelity=0.99, is_operational=True)
    g.add_node(1, T1=10.0, T2=8.0, readout_fidelity=0.95, is_operational=True)
    g.add_node(2, T1=None, T2=None, readout_fidelity=None, is_operational=False)
    g.add_edge(0, 1, gate_fidelity=0.99, gate_time=68.0)
    g.add_edge(1, 2, gate_fidelity=0.90, gate_time=200.0)
    return g


def test_node_score_prefers_higher_t1_t2_readout_by_default():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.node_score(0) > model.node_score(1)


def test_inoperational_qubit_scores_zero():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.node_score(2) == 0.0


def test_maximum_fidelity_objective_uses_readout_only():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="maximum_fidelity")

    # qubit 0 has both better T1/T2 AND better readout, so this doesn't
    # distinguish "readout-only" from "averaged" on its own -- assert the
    # concrete value instead of just an ordering, since maximum_fidelity should
    # equal the min-max normalized readout_fidelity exactly (0.95 -> 0, 0.99 -> 1).
    assert model.node_score(0) == 1.0
    assert model.node_score(1) == 0.0


def test_edge_score_uses_gate_fidelity_by_default():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.edge_score(0, 1) > model.edge_score(1, 2)


def test_minimum_execution_time_objective_favors_faster_gate():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="minimum_execution_time")

    assert model.edge_score(0, 1) > model.edge_score(1, 2)  # 68ns beats 200ns


def test_edge_score_zero_for_nonexistent_edge():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.edge_score(0, 2) == 0.0
