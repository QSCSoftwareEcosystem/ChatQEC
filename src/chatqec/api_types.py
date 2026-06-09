"""Public dataclasses for ChatQEC answers and streams.

Kept separate from `api.py` to avoid circular imports between the public
ChatQEC class (which wires in conversation components) and conversation
modules (which need to construct Citation objects).
"""

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


@dataclass(frozen=True)
class StreamChunk:
    delta: str
