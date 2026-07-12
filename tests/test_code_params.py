"""code_params tool tests (values verified against real Stim output)."""

from chatqec_mcp_tools.tools.code_params import (
    CodeParamsInput,
    CodeParamsOutput,
    code_params,
)


def test_surface_code_params():
    out = code_params(CodeParamsInput(code_family="surface_code_rotated", distance=3))
    assert isinstance(out, CodeParamsOutput)
    assert out.ok is True
    assert (out.n, out.k, out.num_stabilizers) == (9, 1, 8)
    assert out.stim_code_distance == 3
    assert out.num_observables == 1
    assert out.notes == ""


def test_repetition_code_params():
    out = code_params(CodeParamsInput(code_family="repetition_code", distance=5))
    assert out.ok is True
    assert (out.n, out.k, out.num_stabilizers) == (5, 1, 4)
    assert out.stim_code_distance == 5


def test_color_code_reports_graphlike_lower_bound():
    out = code_params(CodeParamsInput(code_family="color_code", distance=3))
    assert out.ok is True
    assert (out.n, out.num_stabilizers) == (7, 6)
    assert out.stim_code_distance == 2  # graphlike distance < true distance 3
    assert out.notes != ""


def test_rejects_even_distance_for_surface():
    out = code_params(CodeParamsInput(code_family="surface_code_rotated", distance=4))
    assert out.ok is False
    assert "odd" in out.error_message.lower()
