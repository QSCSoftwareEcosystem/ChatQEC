"""Coupling-map -> SVG, mapped qubits highlighted.

Follows this repo's existing matplotlib -> SVG-string, best-effort-fallback
pattern (see threshold_sweep._render_plot: "plot is best-effort"). Uses
networkx's own kamada_kawai_layout instead of porting the intern's per-chip
coordinate table (hardware_layout.qubit_coordinates) -- unnecessary for a
"which qubits were picked" diagram.
"""

from __future__ import annotations

import io

import networkx as nx


def render_coupling_diagram(hw_graph: nx.Graph, highlighted_qubits: set[int]) -> str:
    """Render a coupling-map graph as SVG, highlighting specified qubits.

    Args:
        hw_graph: NetworkX graph representing the hardware coupling map.
        highlighted_qubits: Set of qubit indices to highlight in red.

    Returns:
        SVG string representation of the coupling diagram.
        Returns empty string if rendering fails for any reason.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if hw_graph.number_of_nodes() == 0:
            return ""

        pos = nx.kamada_kawai_layout(hw_graph)
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            node_colors = [
                "#d62728" if n in highlighted_qubits else "#cccccc" for n in hw_graph.nodes
            ]
            nx.draw(
                hw_graph,
                pos=pos,
                ax=ax,
                node_color=node_colors,
                with_labels=True,
                node_size=300,
                font_size=8,
            )
            buf = io.StringIO()
            fig.savefig(buf, format="svg")
            return buf.getvalue()
        finally:
            plt.close(fig)
    except Exception:  # plot is best-effort; rendering failure returns empty string
        return ""
