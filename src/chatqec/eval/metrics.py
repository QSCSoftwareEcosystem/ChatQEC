"""Eval metrics: recall@K, citation accuracy, LLM-judged faithfulness."""

from __future__ import annotations

from typing import Any

from chatqec.api import Citation
from chatqec.retrieval.hybrid import RetrievedChunk


def recall_at_k(
    retrieved: list[RetrievedChunk],
    *,
    expected_source_id_prefixes: list[str],
) -> float:
    if not expected_source_id_prefixes:
        return 1.0
    for prefix in expected_source_id_prefixes:
        for chunk in retrieved:
            if chunk.source_id.startswith(prefix):
                return 1.0
    return 0.0


def citation_accuracy(
    citations: list[Citation],
    *,
    expected_source_id_prefixes: list[str],
) -> float:
    if not citations:
        return 0.0
    hits = sum(
        1 for c in citations if any(c.source_id.startswith(p) for p in expected_source_id_prefixes)
    )
    return hits / len(citations)


_JUDGE_SYS = (
    "You are a strict evaluator. Given a question, an answer, and the evidence "
    "the answer was generated from, decide whether the answer's claims follow "
    "from the evidence. Respond with EXACTLY one word: 'yes' or 'no'."
)


def faithfulness_via_judge(
    *,
    question: str,
    answer: str,
    evidence_text: str,
    client: Any,
    model: str,
) -> float:
    user_payload = f"Question: {question}\n\nAnswer: {answer}\n\nEvidence:\n{evidence_text}"
    msg = client.messages.create(
        model=model,
        max_tokens=10,
        system=_JUDGE_SYS,
        messages=[{"role": "user", "content": user_payload}],
    )
    verdict = str(msg.content[0].text).strip().lower()
    return 1.0 if verdict.startswith("yes") else 0.0
