"""Generation pipeline: seed questions -> live chatQEC run -> judge -> candidate
EvalCase rows for human review.

Never writes directly into a real eval set (`eval/qa_set.jsonl` /
`eval/routing_qa_set.jsonl`). Candidates go to a `.candidates.jsonl` file with
`_judge_verdict`/`_routing_flag` hint fields; a human reviews the full
generation log plus those hints and manually promotes approved rows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatqec.api import ChatQEC
from chatqec.eval.judge import judge_and_trim
from chatqec.eval.metrics import numeric_fact_match, tool_call_accuracy, unsafe_tool_invoked

_BANNED_EXPECTED_TOOL = "ibm_submit_execution"


class SeedError(Exception):
    """Raised for a malformed or unsafe seed file."""


@dataclass(frozen=True)
class SeedCase:
    id: str
    subtype: str
    question: str
    complexity: str = "simple"
    expected_tools: list[str] = field(default_factory=list)
    target_capability: list[str] = field(default_factory=list)
    is_adversarial: bool = False
    expected_topics: list[str] = field(default_factory=list)
    expected_source_id_prefixes: list[str] = field(default_factory=list)
    expected_numeric_facts: dict[str, Any] = field(default_factory=dict)


def load_seed_cases(path: Path) -> list[SeedCase]:
    """Load seed cases from a JSONL file. Raises SeedError if any seed's
    expected_tools names a spend-incurring tool that must never be authored as
    "correct" for an automated eval run."""
    out: list[SeedCase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        data = json.loads(line)
        seed = SeedCase(
            id=data["id"],
            subtype=data["subtype"],
            question=data["question"],
            complexity=data.get("complexity", "simple"),
            expected_tools=data.get("expected_tools", []),
            target_capability=data.get("target_capability", []),
            is_adversarial=data.get("is_adversarial", False),
            expected_topics=data.get("expected_topics", []),
            expected_source_id_prefixes=data.get("expected_source_id_prefixes", []),
            expected_numeric_facts=data.get("expected_numeric_facts", {}),
        )
        if _BANNED_EXPECTED_TOOL in seed.expected_tools:
            raise SeedError(
                f"seed {seed.id!r} names {_BANNED_EXPECTED_TOOL!r} in expected_tools — "
                "this tool spends real IBM quota and must never be authored as the "
                "expected/correct routing outcome for an automated eval case."
            )
        out.append(seed)
    return out


def generate_candidates(
    *,
    seeds: list[SeedCase],
    bot: ChatQEC,
    judge_client: Any,
    judge_model: str,
    log_path: Path,
) -> list[dict[str, Any]]:
    """Run each seed through bot.ask(), judge+trim it, append one structured
    record per case to log_path, and return candidate EvalCase-shaped dicts.

    Never writes to a real eval set — the caller/CLI owns the human-review gate.
    """
    candidates: list[dict[str, Any]] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        for seed in seeds:
            t0 = time.monotonic()
            answer = bot.ask(seed.question)
            elapsed_s = time.monotonic() - t0

            base_record: dict[str, Any] = {
                "id": seed.id,
                "timestamp": time.time(),
                "question": seed.question,
                "complexity": seed.complexity,
                "target_capability": seed.target_capability,
                "is_adversarial": seed.is_adversarial,
                "expected_tools": seed.expected_tools,
                "expected_numeric_facts": seed.expected_numeric_facts,
                "raw_answer": answer.text,
                "tool_calls": [
                    {"name": tc.name, "input": tc.input, "output": tc.output}
                    for tc in answer.tool_calls
                ],
                "citations": [c.source_id for c in answer.citations],
                "elapsed_seconds": elapsed_s,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
            }

            if unsafe_tool_invoked(answer.tool_calls):
                log_f.write(
                    json.dumps({**base_record, "verdict": "skipped_unsafe_tool_invoked"}) + "\n"
                )
                continue

            routing_metrics = tool_call_accuracy(
                answer.tool_calls, expected_tools=seed.expected_tools
            )
            numeric_metrics = numeric_fact_match(
                answer.tool_calls, expected=seed.expected_numeric_facts
            )
            evidence_text = "\n".join(f"[{c.source_id}] {c.title}" for c in answer.citations)

            try:
                judge_result = judge_and_trim(
                    question=seed.question,
                    raw_answer=answer.text,
                    tool_calls=answer.tool_calls,
                    evidence_text=evidence_text,
                    expected_tools=seed.expected_tools,
                    client=judge_client,
                    model=judge_model,
                )
            except json.JSONDecodeError as exc:
                log_f.write(
                    json.dumps(
                        {
                            **base_record,
                            "routing_metrics": routing_metrics,
                            "numeric_fact_match": numeric_metrics,
                            "verdict": "judge_parse_error",
                            "judge_parse_error": str(exc),
                        }
                    )
                    + "\n"
                )
                continue

            log_record = {
                **base_record,
                "routing_metrics": routing_metrics,
                "numeric_fact_match": numeric_metrics,
                "judge_verdict": judge_result.verdict,
                "judge_reasoning": judge_result.verdict_reasoning,
                "judge_routing_flag": judge_result.routing_flag,
                "judge_routing_flag_reasoning": judge_result.routing_flag_reasoning,
                "gold_answer_summary": judge_result.gold_answer_summary,
            }
            log_f.write(json.dumps(log_record) + "\n")

            candidates.append(
                {
                    "id": seed.id,
                    "subtype": seed.subtype,
                    "question": seed.question,
                    "expected_topics": seed.expected_topics,
                    "expected_source_id_prefixes": seed.expected_source_id_prefixes,
                    "gold_answer_summary": judge_result.gold_answer_summary,
                    "complexity": seed.complexity,
                    "expected_tools": seed.expected_tools,
                    "target_capability": seed.target_capability,
                    "is_adversarial": seed.is_adversarial,
                    "expected_numeric_facts": seed.expected_numeric_facts,
                    "_judge_verdict": judge_result.verdict,
                    "_routing_flag": judge_result.routing_flag,
                }
            )
    return candidates
