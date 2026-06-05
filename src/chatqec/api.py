"""Public ChatQEC class. Stub in M1; wired up in M7."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    source_id: str
    chunk_id: str
    title: str
    url: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, object]
    output: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    confidence: str = "low"


class ChatQEC:
    """Stub public API. Real wiring lands in Milestone 7."""

    def ask(self, question: str) -> Answer:
        return Answer(
            text=f"[stub] Asked: {question!r}. Synthesis not yet implemented (lands in M7).",
            citations=[],
            tool_calls=[],
            confidence="low",
        )
