"""Eval dataset schema (JSONL) + loader."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class QuestionSubtype(StrEnum):
    DEFINITION = "definition"
    PAPER_LOOKUP = "paper_lookup"
    COMPARISON = "comparison"
    MATH_DETAIL = "math_detail"
    RECENT_NICHE = "recent_niche"
    CODE_EXECUTION = "code_execution"


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    subtype: QuestionSubtype
    question: str
    expected_topics: list[str] = Field(default_factory=list)
    expected_source_id_prefixes: list[str] = Field(default_factory=list)
    gold_answer_summary: str = ""


def load_eval_set(path: Path) -> list[EvalCase]:
    out: list[EvalCase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(EvalCase.model_validate_json(line))
    return out
