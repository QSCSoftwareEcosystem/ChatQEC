from __future__ import annotations

import networkx as nx


def test_render_coupling_diagram_returns_svg_string():
    from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    svg = render_coupling_diagram(g, highlighted_qubits={0, 1})

    assert svg.startswith("<?xml") or "<svg" in svg


def test_render_coupling_diagram_handles_empty_graph_without_raising():
    from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram

    svg = render_coupling_diagram(nx.Graph(), highlighted_qubits=set())

    assert isinstance(svg, str)  # "" on failure, or a valid (if trivial) SVG -- never raises
