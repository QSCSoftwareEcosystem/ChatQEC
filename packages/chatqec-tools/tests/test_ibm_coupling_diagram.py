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


def test_render_coupling_diagram_closes_figure_on_render_failure(monkeypatch):
    """Verify that matplotlib figures are closed even when rendering fails.

    This prevents resource leaks when savefig or draw fails on pathological graphs.
    """
    import matplotlib.pyplot as plt

    from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram

    # Record initial figure count
    initial_fignums = plt.get_fignums()

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    # Monkeypatch fig.savefig to raise an exception
    def failing_savefig(*args, **kwargs):
        raise RuntimeError("Simulated savefig failure")

    def patched_subplots(*args, **kwargs):
        fig, ax = plt.subplots(*args, **kwargs)
        monkeypatch.setattr(fig, "savefig", failing_savefig)
        return fig, ax

    monkeypatch.setattr("matplotlib.pyplot.subplots", patched_subplots)

    # Call render_coupling_diagram with the patched subplots
    svg = render_coupling_diagram(g, highlighted_qubits={0})

    # Verify it returned empty string (graceful failure)
    assert svg == ""

    # Verify no figures leaked: all created figures should be closed
    final_fignums = plt.get_fignums()
    assert final_fignums == initial_fignums, (
        f"Figure leak detected: started with {initial_fignums}, ended with {final_fignums}"
    )
