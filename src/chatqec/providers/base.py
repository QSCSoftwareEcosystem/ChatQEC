"""Shared LLM provider shapes.

The classifier, rewriter, synthesizer, and eval judge all call
``client.messages.create(model, max_tokens, system, messages)`` and expect an
Anthropic-shaped response with ``.content[i].text`` and ``.usage.input_tokens``.

These dataclasses let non-Anthropic providers (HuggingFace, Ollama, ...) speak
the same surface so the rest of the codebase needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    """Anthropic-shaped tool-use request block (for the tool-use loop)."""

    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class MessageResponse:
    """Anthropic-shaped response object."""

    content: list[TextBlock | ToolUseBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
