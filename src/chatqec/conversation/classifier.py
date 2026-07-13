"""Off-topic classifier (Haiku).

Cheap pre-filter that refuses non-QEC queries before reaching Sonnet.
Saves cost and adds an abuse guard (spec §12.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chatqec.trace import trace

_SYS = (
    "You are a strict classifier. Decide if the user's question is about "
    "Quantum Error Correction (QEC), fault-tolerant quantum computation, "
    "quantum codes, decoders, or directly related topics (stabilizer formalism, "
    "magic state distillation, surface code, etc.). "
    "Respond with EXACTLY one word and nothing else: 'on_topic' or 'off_topic'. "
    "Do NOT answer the question or add any text after the label. "
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
        trace("classifier.call", model=self.model, query=query)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system=_SYS,
            messages=[{"role": "user", "content": query}],
        )
        text = str(msg.content[0].text).strip().lower()
        trace("classifier.response", model=self.model, raw=text)
        # Key off the leading label token, not exact-match: Haiku sometimes emits
        # the label and then keeps answering (e.g. "on_topic\n\nThe Steane code is …"
        # when the query says "give me an example"). Strip surrounding punctuation so
        # "on_topic." / "on_topic:" also count. A truly malformed first token
        # (anything but "on_topic") stays off_topic, which is the safe default.
        tokens = text.split()
        first = tokens[0].strip(".,:;!?\"'`") if tokens else ""
        if first == "on_topic":
            return TopicVerdict(on_topic=True)
        return TopicVerdict(on_topic=False, refusal_message=_DEFAULT_REFUSAL)
