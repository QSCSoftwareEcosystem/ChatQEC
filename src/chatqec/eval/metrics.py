"""Eval metrics: recall@K, citation accuracy, LLM-judged faithfulness,
tool-routing accuracy, and judge-independent numeric-fact checks."""

from __future__ import annotations

import ast
import json
from typing import Any

from chatqec.api import Citation, ToolCall
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


def tool_call_accuracy(
    tool_calls: list[ToolCall],
    *,
    expected_tools: list[str],
) -> dict[str, float]:
    """Set-based precision/recall/F1 over invoked vs. expected LLM-routed tool
    names. Order-independent: a compound/complex case may legitimately invoke
    its tools in either order.

    Returns {} when expected_tools is empty — not applicable (a deterministic-
    routed case, e.g. wiki/web, or a case where no tool should fire at all —
    mirrors how citation_accuracy degenerates on an empty expected list).
    """
    if not expected_tools:
        return {}
    invoked = {tc.name for tc in tool_calls}
    expected = set(expected_tools)
    tp = len(invoked & expected)
    precision = tp / len(invoked) if invoked else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": 1.0 if invoked == expected else 0.0,
        "n_invoked": float(len(tool_calls)),
        "n_expected": float(len(expected)),
    }


def unsafe_tool_invoked(
    tool_calls: list[ToolCall],
    *,
    banned: tuple[str, ...] = ("ibm_submit_execution",),
) -> bool:
    """Did the run invoke a spend-incurring tool that must never fire in eval?"""
    return any(tc.name in banned for tc in tool_calls)


def parse_tool_output(text: str) -> dict[str, Any]:
    """A tool call's recorded output as a dict, or {} when it is not one.

    api.py records ``str(result_dict)``, a Python repr (single quotes, True/None), which
    json.loads rejects; the Chat page reads it with ast.literal_eval. Accept both, so JSON
    outputs (tests, other callers) keep working. literal_eval never executes code.
    """
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(text)
        except (TypeError, ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def numeric_fact_match(
    tool_calls: list[ToolCall],
    *,
    expected: dict[str, float | int | str],
) -> dict[str, bool]:
    """Judge-independent correctness check: compare expected numeric/structural
    facts (e.g. {"n": 72, "k": 12} for a code_params question) directly against
    parsed tool-call output.

    Motivated by OmniQEC (arXiv:2607.25865): the authors reject LLM-as-a-judge
    for factual/numeric claims, grounding correctness in executable tool
    output instead. Each tool's JSON output is parsed and unioned; a key in
    `expected` is looked up across that union, so the caller just needs to
    name the field the relevant tool actually returns (e.g. `code_params`
    returns `n`/`k`/`stim_code_distance`; `qec_circuit_build` returns
    `num_qubits`/`num_detectors`/`num_observables` — there is no single
    canonical schema across tools).

    Returns {} when `expected` is empty — not applicable.
    """
    if not expected:
        return {}
    observed: dict[str, Any] = {}
    for tc in tool_calls:
        observed.update(parse_tool_output(tc.output))
    return {key: (key in observed and observed[key] == want) for key, want in expected.items()}
