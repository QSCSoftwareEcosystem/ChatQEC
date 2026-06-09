"""Off-topic classifier (Haiku).

Cheap pre-filter that refuses non-QEC queries before reaching Sonnet.
Saves cost and adds an abuse guard (spec §12.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SYS = (
    "You are a strict classifier. Decide if the user's question is about "
    "Quantum Error Correction (QEC), fault-tolerant quantum computation, "
    "quantum codes, decoders, or directly related topics (stabilizer formalism, "
    "magic state distillation, surface code, etc.). "
    "Respond with EXACTLY one word: 'on_topic' or 'off_topic'. "
    "Borderline / general quantum computing questions that don't involve QEC: off_topic."
)

_DEFAULT_REFUSAL = (
    "ChatQEC only answers Quantum Error Correction questions. "
    "Please rephrase your question to focus on QEC, fault tolerance, "
    "stabilizer codes, decoders, or related topics."
)


@dataclass(frozen=True)
class TopicVerdict:
    on_topic: bool
    refusal_message: str = ""


class OffTopicClassifier:
    """Wraps an Anthropic client to perform single-token on/off-topic classification.

    The ``client`` parameter is typed as ``Any`` to avoid pinning the test surface to
    the full Anthropic SDK type; in production it is an ``anthropic.Anthropic`` instance.
    """

    def __init__(self, *, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def classify(self, query: str) -> TopicVerdict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system=_SYS,
            messages=[{"role": "user", "content": query}],
        )
        text = str(msg.content[0].text).strip().lower()
        if text == "on_topic":
            return TopicVerdict(on_topic=True)
        return TopicVerdict(on_topic=False, refusal_message=_DEFAULT_REFUSAL)
