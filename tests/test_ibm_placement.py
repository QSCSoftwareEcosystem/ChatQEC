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
