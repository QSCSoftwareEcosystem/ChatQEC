"""stim_diagram tool tests."""

from chatqec_mcp_tools.tools.stim_diagram import (
    StimDiagramInput,
    StimDiagramOutput,
    stim_diagram,
)


def test_diagram_returns_svg_string_for_valid_circuit():
    out = stim_diagram(StimDiagramInput(circuit="H 0\nCNOT 0 1\nM 0 1\n"))
    assert isinstance(out, StimDiagramOutput)
    assert out.ok is True
    assert out.svg.lstrip().startswith("<")  # SVG begins with '<' element


def test_diagram_returns_error_for_invalid_circuit():
    out = stim_diagram(StimDiagramInput(circuit="NOT_A_GATE 0\n"))
    assert out.ok is False
    assert "error" in out.error_message.lower()
