"""Query rewriter (Haiku).

Reads recent conversation history and produces:
  - rewritten: a self-contained query (pronouns/refs resolved)
  - variants: 0-3 alternative phrasings to widen retrieval recall
  - is_followup: signals the synthesizer to reuse previous turn context
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from chatqec.conversation.memory import ConversationMemory
from chatqec.trace import trace

_SYS = """You are a query rewriter for a Quantum Error Correction Q&A assistant.

Given the user's most recent question and (optionally) the recent conversation,
rewrite the question to be:
1. Self-contained (resolve pronouns like "it", "this", "that")
2. Specific (expand QEC abbreviations: MWPM, BP-OSD, GKP, MSD, etc.)
3. Searchable (use canonical terminology)

Also produce up to 2 alternative phrasings ("variants") to widen retrieval recall.

Set is_followup=true if the question depends on prior turns.

Respond with ONLY a JSON object of the form:
{"rewritten": "...", "variants": ["...", "..."], "is_followup": true|false}
"""


@dataclass(frozen=True)
class RewriteResult:
    rewritten: str
    variants: list[str] = field(default_factory=list)
    is_followup: bool = False


class QueryRewriter:
    """Wraps an Anthropic client to rewrite the latest user query in context.

    ``client`` is typed ``Any`` to avoid pinning to the full Anthropic SDK type;
    in production it is an ``anthropic.Anthropic`` instance.
    """

    def __init__(self, *, client: Any, model: str, history_turns: int = 6) -> None:
        self.client = client
        self.model = model
        self.history_turns = history_turns

    def rewrite(self, query: str, *, memory: ConversationMemory) -> RewriteResult:
        recent = memory.recent(n=self.history_turns)
        history_block = "\n".join(f"{t.role}: {t.content}" for t in recent)
        user_payload = (
            f"Recent conversation:\n{history_block or '(none)'}\n\n"
            f"Most recent question: {query}\n\nReturn the JSON."
        )
        trace("rewriter.call", model=self.model, query=query, history_turns=len(recent))
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=_SYS,
            messages=[{"role": "user", "content": user_payload}],
        )
        raw = str(msg.content[0].text).strip()
        trace("rewriter.response", model=self.model, raw=raw)
        try:
            parsed = json.loads(raw)
            return RewriteResult(
                rewritten=str(parsed.get("rewritten", query)),
                variants=[str(v) for v in parsed.get("variants", [])][:3],
                is_followup=bool(parsed.get("is_followup", False)),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return RewriteResult(rewritten=query)
