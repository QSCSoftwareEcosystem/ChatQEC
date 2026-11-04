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
    figure_paths: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # Sources of every chunk the synthesizer was given (canonical, retrieved, web), in
    # order and de-duplicated: what retrieval delivered, whether or not it was cited.
    retrieved_source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamChunk:
    delta: str


@dataclass(frozen=True)
class ImageAttachment:
    data: bytes
    media_type: str
    filename: str = ""
