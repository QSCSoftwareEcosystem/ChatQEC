"""Eval runner — executes a case list against a ChatQEC instance and aggregates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chatqec.api import ChatQEC
from chatqec.eval.dataset import EvalCase
from chatqec.eval.metrics import (
    citation_accuracy,
    faithfulness_via_judge,
    numeric_fact_match,
    recall_at_k,
    tool_call_accuracy,
)
from chatqec.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class SubtypeStats:
    n: int
    recall_at_k: float
    citation_accuracy: float
    faithfulness: float


@dataclass(frozen=True)
class BucketStats:
    n: int
    recall_at_k: float
    citation_accuracy: float
    faithfulness: float
    tool_call_f1: float | None  # None when no case in the bucket has expected_tools


@dataclass(frozen=True)
class EvalReport:
    total: int
    overall_recall_at_k: float
    overall_citation_accuracy: float
    overall_faithfulness: float
    per_subtype: dict[str, SubtypeStats] = field(default_factory=dict)
    per_complexity: dict[str, BucketStats] = field(default_factory=dict)
    per_capability: dict[str, BucketStats] = field(default_factory=dict)
    adversarial_vs_non: dict[str, BucketStats] = field(default_factory=dict)
    raw: list[dict[str, Any]] = field(default_factory=list)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_f1(rows: list[dict[str, Any]]) -> float | None:
    f1s = [float(r["tool_call_accuracy"]["f1"]) for r in rows if r["tool_call_accuracy"]]
    return _mean(f1s) if f1s else None


def _bucket_stats(rows: list[dict[str, Any]]) -> BucketStats:
    return BucketStats(
        n=len(rows),
        recall_at_k=_mean([float(r["recall_at_k"]) for r in rows]),
        citation_accuracy=_mean([float(r["citation_accuracy"]) for r in rows]),
        faithfulness=_mean([float(r["faithfulness"]) for r in rows]),
        tool_call_f1=_mean_f1(rows),
    )


def run_eval(
    *,
    cases: list[EvalCase],
    bot: ChatQEC,
    judge_client: Any,
    judge_model: str,
) -> EvalReport:
    raw_rows: list[dict[str, Any]] = []
    by_subtype: dict[str, list[dict[str, Any]]] = {}
    by_complexity: dict[str, list[dict[str, Any]]] = {}
    by_capability: dict[str, list[dict[str, Any]]] = {}
    by_adversarial: dict[str, list[dict[str, Any]]] = {"adversarial": [], "non_adversarial": []}

    for case in cases:
        t0 = time.monotonic()
        answer = bot.ask(case.question)
        elapsed_s = time.monotonic() - t0
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
        routing = tool_call_accuracy(answer.tool_calls, expected_tools=case.expected_tools)
        numeric = numeric_fact_match(answer.tool_calls, expected=case.expected_numeric_facts)
        row: dict[str, Any] = {
            "id": case.id,
            "subtype": case.subtype.value,
            "complexity": case.complexity,
            "target_capability": case.target_capability,
            "is_adversarial": case.is_adversarial,
            "recall_at_k": r,
            "citation_accuracy": ca,
            "faithfulness": faith,
            "answer": answer.text,
            "citation_count": len(answer.citations),
            "tool_calls": [{"name": tc.name, "input": tc.input} for tc in answer.tool_calls],
            "elapsed_seconds": elapsed_s,
            "tool_call_accuracy": routing,
            "numeric_fact_match": numeric,
        }
        raw_rows.append(row)
        by_subtype.setdefault(case.subtype.value, []).append(row)
        by_complexity.setdefault(case.complexity, []).append(row)
        for cap in case.target_capability:
            by_capability.setdefault(cap, []).append(row)
        by_adversarial["adversarial" if case.is_adversarial else "non_adversarial"].append(row)

    per_subtype = {
        k: SubtypeStats(
            n=len(rows),
            recall_at_k=_mean([float(r["recall_at_k"]) for r in rows]),
            citation_accuracy=_mean([float(r["citation_accuracy"]) for r in rows]),
            faithfulness=_mean([float(r["faithfulness"]) for r in rows]),
        )
        for k, rows in by_subtype.items()
    }
    per_complexity = {k: _bucket_stats(rows) for k, rows in by_complexity.items()}
    per_capability = {k: _bucket_stats(rows) for k, rows in by_capability.items()}
    adversarial_vs_non = {k: _bucket_stats(rows) for k, rows in by_adversarial.items() if rows}

    return EvalReport(
        total=len(cases),
        overall_recall_at_k=_mean([float(r["recall_at_k"]) for r in raw_rows]),
        overall_citation_accuracy=_mean([float(r["citation_accuracy"]) for r in raw_rows]),
        overall_faithfulness=_mean([float(r["faithfulness"]) for r in raw_rows]),
        per_subtype=per_subtype,
        per_complexity=per_complexity,
        per_capability=per_capability,
        adversarial_vs_non=adversarial_vs_non,
        raw=raw_rows,
    )
