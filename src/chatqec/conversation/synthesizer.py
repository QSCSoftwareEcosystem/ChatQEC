"""Synthesizer — wraps Anthropic Sonnet for non-streaming and streaming answers.

M7 has no tool use; M8 adds the Stim tool-use loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from chatqec.api_types import Citation, ImageAttachment
from chatqec.conversation.citations import extract_citations, strip_trailing_sources
from chatqec.conversation.image_blocks import anthropic_image_block
from chatqec.conversation.mcp_adapter import MCPAdapter
from chatqec.conversation.memory import Turn
from chatqec.conversation.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    assemble_context_block,
)
from chatqec.conversation.tool_loop import ToolInvocation, run_tool_loop
from chatqec.retrieval.hybrid import RetrievedChunk
from chatqec.trace import trace


@dataclass(frozen=True)
class SynthesisResult:
    text: str
    citations: list[Citation]
    confidence: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_invocations: list[ToolInvocation] = field(default_factory=list)


_HIGH_CONFIDENCE_CITATIONS = 3
_MIN_CITATIONS_FOR_MEDIUM = 1


def _confidence_from_citations(n_citations: int, *, has_context: bool) -> str:
    if not has_context:
        return "low"
    if n_citations >= _HIGH_CONFIDENCE_CITATIONS:
        return "high"
    if n_citations >= _MIN_CITATIONS_FOR_MEDIUM:
        return "medium"
    return "low"


class Synthesizer:
    """Wraps Anthropic Sonnet. ``client`` typed ``Any`` to avoid SDK pinning."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 4000,
        mcp_adapter: MCPAdapter | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.mcp_adapter = mcp_adapter

    def _build_messages(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
        images: list[ImageAttachment] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": t.role, "content": t.content} for t in history]
        context = assemble_context_block(retrieved, canonical_chunks=canonical_chunks)
        text = f"{context}\n\nQuestion: {question}"
        if images:
            content: list[dict[str, Any]] = [anthropic_image_block(i) for i in images]
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": text})
        return messages

    def synthesize(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
        images: list[ImageAttachment] | None = None,
    ) -> SynthesisResult:
        messages: list[dict[str, Any]] = list(
            self._build_messages(
                question=question,
                retrieved=retrieved,
                canonical_chunks=canonical_chunks,
                history=history,
                images=images,
            )
        )

        trace(
            "synthesizer.call",
            model=self.model,
            mcp_enabled=self.mcp_adapter is not None,
            n_retrieved=len(retrieved),
            n_canonical=len(canonical_chunks),
            n_history=len(history),
        )
        invocations: list[ToolInvocation] = []
        if self.mcp_adapter is not None:
            text, invocations = run_tool_loop(
                client=self.client,
                model=self.model,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                messages=messages,
                adapter=self.mcp_adapter,
                max_iterations=5,
                max_tokens=self.max_tokens,
            )
            usage_in = 0
            usage_out = 0
        else:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                messages=messages,
            )
            # Join all text blocks; tolerate a response with no text content
            # (some providers return an empty content list) instead of crashing.
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            usage = getattr(msg, "usage", None)
            usage_in = getattr(usage, "input_tokens", 0) if usage else 0
            usage_out = getattr(usage, "output_tokens", 0) if usage else 0

        all_context = canonical_chunks + retrieved
        # Extract citations from the full text (incl. any model "Sources:" list)
        # to score confidence, then strip that list so the app renders only its
        # own linked source block — no duplicate.
        citations = extract_citations(text, all_context)
        return SynthesisResult(
            text=strip_trailing_sources(text),
            citations=citations,
            confidence=_confidence_from_citations(len(citations), has_context=bool(all_context)),
            input_tokens=usage_in,
            output_tokens=usage_out,
            tool_invocations=invocations,
        )

    def synthesize_stream(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
        images: list[ImageAttachment] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive.

        Tool use can't stream token-by-token, and some clients (the fallback
        facade) don't implement ``stream`` at all — in both cases we produce the
        full answer and yield it as a single chunk instead of crashing.
        """
        messages = self._build_messages(
            question=question,
            retrieved=retrieved,
            canonical_chunks=canonical_chunks,
            history=history,
            images=images,
        )

        trace(
            "synthesizer.stream_call",
            model=self.model,
            mcp_enabled=self.mcp_adapter is not None,
            n_retrieved=len(retrieved),
            n_canonical=len(canonical_chunks),
            n_history=len(history),
        )
        # Tool-enabled answers run the (non-streaming) tool loop.
        if self.mcp_adapter is not None:
            text, _ = run_tool_loop(
                client=self.client,
                model=self.model,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                messages=messages,
                adapter=self.mcp_adapter,
                max_tokens=self.max_tokens,
            )
            yield strip_trailing_sources(str(text))
            return

        # Clients without streaming support fall back to a single create().
        stream_fn = getattr(self.client.messages, "stream", None)
        if stream_fn is None:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYNTHESIZER_SYSTEM_PROMPT,
                messages=messages,
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            yield strip_trailing_sources(str(text))
            return

        with stream_fn(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                yield str(text_chunk)
