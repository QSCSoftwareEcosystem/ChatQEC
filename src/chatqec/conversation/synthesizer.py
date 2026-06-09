"""Synthesizer — wraps Anthropic Sonnet for non-streaming and streaming answers.

M7 has no tool use; M8 adds the Stim tool-use loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from chatqec.api import Citation
from chatqec.conversation.citations import extract_citations
from chatqec.conversation.memory import Turn
from chatqec.conversation.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    assemble_context_block,
)
from chatqec.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class SynthesisResult:
    text: str
    citations: list[Citation]
    confidence: str
    input_tokens: int = 0
    output_tokens: int = 0


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

    def __init__(self, *, client: Any, model: str, max_tokens: int = 4000) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _build_messages(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": t.role, "content": t.content} for t in history]
        context = assemble_context_block(retrieved, canonical_chunks=canonical_chunks)
        messages.append(
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}",
            }
        )
        return messages

    def synthesize(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
    ) -> SynthesisResult:
        messages = self._build_messages(
            question=question,
            retrieved=retrieved,
            canonical_chunks=canonical_chunks,
            history=history,
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            messages=messages,
        )
        text = str(msg.content[0].text)
        all_context = canonical_chunks + retrieved
        citations = extract_citations(text, all_context)
        usage = getattr(msg, "usage", None)
        return SynthesisResult(
            text=text,
            citations=citations,
            confidence=_confidence_from_citations(len(citations), has_context=bool(all_context)),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    def synthesize_stream(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
    ) -> Iterator[str]:
        """Yield text deltas as they arrive."""
        messages = self._build_messages(
            question=question,
            retrieved=retrieved,
            canonical_chunks=canonical_chunks,
            history=history,
        )
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                yield str(text_chunk)
