"""Eval-set schema + loader tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chatqec.eval.dataset import EvalCase, QuestionSubtype, load_eval_set


def test_eval_case_required_fields():
    c = EvalCase(
        id="def-001",
        subtype=QuestionSubtype.DEFINITION,
        question="What is the surface code?",
        expected_topics=["surface-code"],
        expected_source_id_prefixes=["canonical:surface-code"],
        gold_answer_summary="A topological CSS code on a square lattice.",
    )
    assert c.id == "def-001"


def test_eval_case_rejects_unknown_subtype():
    with pytest.raises(ValidationError):
        EvalCase(
            id="x",
            subtype="invalid",  # type: ignore[arg-type]
            question="q",
            expected_topics=[],
            expected_source_id_prefixes=[],
            gold_answer_summary="",
        )


def test_load_eval_set_round_trips(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        '{"id":"d-1","subtype":"definition","question":"q","expected_topics":["t"],'
        '"expected_source_id_prefixes":["canonical:t"],'
        '"gold_answer_summary":"s"}\n',
        encoding="utf-8",
    )
    cases = load_eval_set(p)
    assert len(cases) == 1
    assert cases[0].id == "d-1"


def test_load_eval_set_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        '{"id":"d-1","subtype":"definition","question":"q","expected_topics":["t"],'
        '"expected_source_id_prefixes":["canonical:t"],'
        '"gold_answer_summary":"s"}\n\n\n',
        encoding="utf-8",
    )
    assert len(load_eval_set(p)) == 1


def test_subtype_enum_covers_six_classes():
    expected = {
        "definition",
        "paper_lookup",
        "comparison",
        "math_detail",
        "recent_niche",
        "code_execution",
    }
    assert {s.value for s in QuestionSubtype} == expected
