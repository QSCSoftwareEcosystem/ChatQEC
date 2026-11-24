"""code_params tool tests (values verified against real Stim output)."""

import pytest
from pydantic import ValidationError

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


# --- literature families: values copied from papers, not computed -------------------------


def test_bb5_returns_every_row_with_its_source_and_location():
    out = code_params(CodeParamsInput(code_family="bb5"))
    assert out.ok, out.error_message
    triples = {(r["n"], r["k"], r["d"]) for r in out.rows}
    assert {(30, 4, 5), (48, 4, 7), (18, 4, 3), (24, 4, 4)} <= triples
    assert all(r["source"].startswith("arXiv:") and r["locator"] for r in out.rows)
    assert "arXiv:2503.22071" in out.source


def test_distance_filters_literature_rows():
    out = code_params(CodeParamsInput(code_family="tile", distance=19))
    assert out.ok and [(r["n"], r["k"], r["d"]) for r in out.rows] == [(512, 18, 19)]


def test_a_distance_with_no_row_lists_what_the_papers_report():
    out = code_params(CodeParamsInput(code_family="tile", distance=3))
    assert out.ok is False
    assert "12" in out.error_message and "19" in out.error_message


def test_literature_families_do_not_claim_a_computed_distance():
    out = code_params(CodeParamsInput(code_family="barbell", distance=12))
    assert out.ok and out.stim_code_distance == 0
    assert "papers" in out.notes and "not computed" in out.notes


def test_stim_families_still_require_a_distance():
    with pytest.raises(ValidationError, match="distance"):
        CodeParamsInput(code_family="surface_code_rotated")


def test_all_four_literature_families_are_accepted():
    for family in ("bb5", "tile", "directional_tile", "barbell"):
        assert code_params(CodeParamsInput(code_family=family)).ok
