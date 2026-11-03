"""LLM-judge used at eval-case *generation* time: verifies correctness of a raw
answer against the tool/evidence it was generated from, trims it into a
concise gold_answer_summary, and flags possible routing mismatches for human
review.

Distinct from `faithfulness_via_judge` in `metrics.py`, which is a binary
scoring judge run on every `eval run`. This one runs once per case, at
authoring time, returns structured multi-field output, and its output is
reviewed by a human before a case is accepted into an eval set (see
`eval/generate.py`).

Per OmniQEC (arXiv:2607.25865), this judge is deliberately NOT the ground
truth for tool-routing correctness (the seed author's `expected_tools` is)
and NOT the primary correctness signal for numeric/structural facts (see
`metrics.numeric_fact_match`, checked directly against tool output instead).
It is scoped to what an LLM judge is actually suited for: summarizing prose
and flagging things for a human to look at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chatqec.api import ToolCall

_GENERATION_JUDGE_SYS = """You are assisting with the construction of a gold-standard
evaluation set for a QEC research assistant. You will be given a question, the
assistant's raw answer, the tool calls it made (name/input/output) and/or the
retrieved evidence, and the tool names the question's author expected to be used.

Do three things:
1. Judge whether the raw answer's factual claims are correct and supported by the
   tool output / evidence shown (not by your own outside knowledge of QEC).
2. Produce a clipped, concise gold_answer_summary (2-4 sentences) - the minimum
   text needed to check a future answer against, stripped of hedging, repeated
   caveats, or restated tool output.
3. Compare the tools actually invoked to the author's expected_tools list, and
   note ONLY whether they look mismatched (you are not the ground truth for
   routing correctness - the author's expected_tools is authoritative; you are
   flagging a possible authoring error or a possible genuine misroute for a human
   to review).

Respond with ONLY a JSON object, no prose outside it:
{
  "verdict": "correct" | "incorrect" | "partially_correct",
  "verdict_reasoning": "<1-3 sentences citing specific evidence/tool output>",
  "gold_answer_summary": "<clipped answer>",
  "routing_flag": "match" | "mismatch" | "not_applicable",
  "routing_flag_reasoning": "<1 sentence>"
}
"""


@dataclass(frozen=True)
class GenerationJudgeResult:
    verdict: str
    verdict_reasoning: str
    gold_answer_summary: str
    routing_flag: str
    routing_flag_reasoning: str


def judge_and_trim(
    *,
    question: str,
    raw_answer: str,
    tool_calls: list[ToolCall],
    evidence_text: str,
    expected_tools: list[str],
    client: Any,
    model: str,
) -> GenerationJudgeResult:
    """Runs the generation-time judge. Raises json.JSONDecodeError on a
    malformed judge response — the caller (eval/generate.py) is responsible
    for catching this and logging it as a `judge_parse_error` rather than
    aborting a whole batch."""
    tool_call_text = (
        "\n".join(
            f"- {tc.name}(input={json.dumps(tc.input)}) -> output={tc.output[:2000]}"
            for tc in tool_calls
        )
        or "(no tools invoked)"
    )
    user_payload = (
        f"Question: {question}\n\n"
        f"Raw answer: {raw_answer}\n\n"
        f"Tool calls made:\n{tool_call_text}\n\n"
        f"Retrieved evidence:\n{evidence_text or '(none)'}\n\n"
        f"Author-expected tools: {expected_tools or '(none - deterministic route or no tool expected)'}"
    )
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        system=_GENERATION_JUDGE_SYS,
        messages=[{"role": "user", "content": user_payload}],
    )
    raw = str(msg.content[0].text).strip()
    data = json.loads(raw)
    return GenerationJudgeResult(
        verdict=data["verdict"],
        verdict_reasoning=data["verdict_reasoning"],
        gold_answer_summary=data["gold_answer_summary"],
        routing_flag=data["routing_flag"],
        routing_flag_reasoning=data["routing_flag_reasoning"],
    )
