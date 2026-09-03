from __future__ import annotations

import networkx as nx


def _line_hardware_graph(n: int):
    """0-1-2-3-...-n-1 line, uniform quality except qubit 0 is best."""
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=100.0 if q == 0 else 50.0, T2=90.0 if q == 0 else 45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=50.0)
    return g


def test_place_greedy_returns_valid_mapping_for_ghz():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)  # GHZ-3 star: logical 0 talks to 1 and 2

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert result.strategy_used == "greedy"
    assert set(result.assignment.keys()) == {0, 1, 2}
    assert len(set(result.assignment.values())) == 3  # no physical qubit reused


def test_place_greedy_seeds_on_highest_quality_qubit():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=5)  # logical 0 is the most-interacting node

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert result.assignment[0] == 0  # physical qubit 0 is the best-quality seed


def test_place_greedy_handles_isolated_logical_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)
    interaction = nx.Graph()
    interaction.add_nodes_from([0, 1, 2])  # no edges -- e.g. three idle spectator qubits

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert len(result.assignment) == 3
    assert len(set(result.assignment.values())) == 3


def test_place_vf2_finds_exact_match_when_one_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)  # 0-1-2-3-4
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(1, 2, weight=1)  # a 3-node path -- exists as a subgraph of the line

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality)

    assert result is not None
    assert result.strategy_used == "vf2"
    pu, pv = result.assignment[0], result.assignment[1]
    pv2 = result.assignment[2]
    assert hw_graph.has_edge(pu, pv)
    assert hw_graph.has_edge(pv, pv2)


def test_place_vf2_returns_none_when_no_exact_match_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)  # 0-1-2, max degree 2
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)
    interaction.add_edge(1, 2, weight=1)  # a triangle -- no triangle exists in a line graph

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality)

    assert result is None


def test_place_vf2_respects_zero_time_budget_by_returning_none():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality, time_budget_seconds=0.0)

    assert result is None


def test_place_dispatcher_falls_back_to_greedy_when_vf2_has_no_match():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)
    interaction.add_edge(1, 2, weight=1)  # triangle, impossible on a line

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place(hw_graph, interaction, quality)

    assert result.strategy_used == "greedy"
    assert len(result.assignment) == 3


def test_place_dispatcher_prefers_vf2_when_exact_match_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place(hw_graph, interaction, quality)

    assert result.strategy_used == "vf2"
