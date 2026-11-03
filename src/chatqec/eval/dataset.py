"""Eval dataset schema (JSONL) + loader."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Complexity = Literal["simple", "compound", "complex"]


class QuestionSubtype(StrEnum):
    DEFINITION = "definition"
    PAPER_LOOKUP = "paper_lookup"
    COMPARISON = "comparison"
    MATH_DETAIL = "math_detail"
    RECENT_NICHE = "recent_niche"
    CODE_EXECUTION = "code_execution"
    HARDWARE_MAPPING = "hardware_mapping"


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    subtype: QuestionSubtype
    question: str
    expected_topics: list[str] = Field(default_factory=list)
    expected_source_id_prefixes: list[str] = Field(default_factory=list)
    gold_answer_summary: str = ""

    # --- tool-routing eval fields ---
    complexity: Complexity = "simple"
    expected_tools: list[str] = Field(default_factory=list)
    # Free-form, not a StrEnum: new tools land in the sibling chatqec-mcp-tools
    # repo on its own release cycle. Convention (not enforced): "stim", "ibm",
    # "qappswiki", "websearch".
    target_capability: list[str] = Field(default_factory=list)
    is_adversarial: bool = False
    # Numeric/structural facts checked directly against tool output, independent
    # of the LLM judge (e.g. {"n": 72, "k": 12, "d": 6} for a code_params case).
    expected_numeric_facts: dict[str, float | int | str] = Field(default_factory=dict)


def load_eval_set(path: Path) -> list[EvalCase]:
    out: list[EvalCase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(EvalCase.model_validate_json(line))
    return out
