"""Eval runner — executes a case list against a ChatQEC instance and aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chatqec.api import ChatQEC
from chatqec.eval.dataset import EvalCase
from chatqec.eval.metrics import (
    citation_accuracy,
    faithfulness_via_judge,
    recall_at_k,
)
from chatqec.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class SubtypeStats:
    n: int
    recall_at_k: float
    citation_accuracy: float
    faithfulness: float


@dataclass(frozen=True)
class EvalReport:
    total: int
    overall_recall_at_k: float
    overall_citation_accuracy: float
    overall_faithfulness: float
    per_subtype: dict[str, SubtypeStats] = field(default_factory=dict)
    raw: list[dict[str, Any]] = field(default_factory=list)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_eval(
    *,
    cases: list[EvalCase],
    bot: ChatQEC,
    judge_client: Any,
    judge_model: str,
) -> EvalReport:
    raw_rows: list[dict[str, Any]] = []
    by_subtype: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        answer = bot.ask(case.question)
        approximated_retrieved = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                source_id=c.source_id,
                text="",
                title=c.title,
                authority_tier="",
                score=0.0,
                payload={},
            )
            for c in answer.citations
        ]
        r = recall_at_k(
            approximated_retrieved,
            expected_source_id_prefixes=case.expected_source_id_prefixes,
        )
        ca = citation_accuracy(
            answer.citations,
            expected_source_id_prefixes=case.expected_source_id_prefixes,
        )
        evidence_text = "\n".join(f"[{c.source_id}] {c.title}" for c in answer.citations)
        faith = faithfulness_via_judge(
            question=case.question,
            answer=answer.text,
            evidence_text=evidence_text,
            client=judge_client,
            model=judge_model,
        )
        row: dict[str, Any] = {
            "id": case.id,
            "subtype": case.subtype.value,
            "recall_at_k": r,
            "citation_accuracy": ca,
            "faithfulness": faith,
            "answer": answer.text,
            "citation_count": len(answer.citations),
        }
        raw_rows.append(row)
        by_subtype.setdefault(case.subtype.value, []).append(row)

    per_subtype = {
        k: SubtypeStats(
            n=len(rows),
            recall_at_k=_mean([float(r["recall_at_k"]) for r in rows]),
            citation_accuracy=_mean([float(r["citation_accuracy"]) for r in rows]),
            faithfulness=_mean([float(r["faithfulness"]) for r in rows]),
        )
        for k, rows in by_subtype.items()
    }

    return EvalReport(
        total=len(cases),
        overall_recall_at_k=_mean([float(r["recall_at_k"]) for r in raw_rows]),
        overall_citation_accuracy=_mean([float(r["citation_accuracy"]) for r in raw_rows]),
        overall_faithfulness=_mean([float(r["faithfulness"]) for r in raw_rows]),
        per_subtype=per_subtype,
        raw=raw_rows,
    )
