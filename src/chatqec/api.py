"""ChatQEC public API (replaces the M1 stub).

End-to-end: classifier -> rewriter -> retriever -> synthesizer.
"""

from __future__ import annotations

from collections.abc import Iterator

from chatqec.api_types import Answer, Citation, StreamChunk, ToolCall
from chatqec.conversation.classifier import OffTopicClassifier
from chatqec.conversation.memory import ConversationMemory
from chatqec.conversation.rewriter import QueryRewriter
from chatqec.conversation.synthesizer import Synthesizer
from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.hybrid import HybridRetriever

__all__ = [
    "Answer",
    "ChatQEC",
    "Citation",
    "StreamChunk",
    "ToolCall",
]

_HISTORY_TURNS = 6


class ChatQEC:
    """End-to-end QEC research assistant."""

    def __init__(
        self,
        *,
        classifier: OffTopicClassifier,
        rewriter: QueryRewriter,
        retriever: HybridRetriever,
        synthesizer: Synthesizer,
        canonical_index: CanonicalIndex,
        memory: ConversationMemory | None = None,
        top_k: int = 8,
        candidate_k: int = 50,
    ) -> None:
        self.classifier = classifier
        self.rewriter = rewriter
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.canonical_index = canonical_index
        self.memory = memory or ConversationMemory()
        self.top_k = top_k
        self.candidate_k = candidate_k

    def ask(self, question: str) -> Answer:
        verdict = self.classifier.classify(question)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            return Answer(text=verdict.refusal_message, confidence="low")

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]

        retrieval = self.retriever.retrieve(
            queries=queries, top_k=self.top_k, candidate_k=self.candidate_k
        )
        result = self.synthesizer.synthesize(
            question=question,
            retrieved=retrieval.chunks,
            canonical_chunks=[],
            history=self.memory.recent(n=_HISTORY_TURNS),
        )

        self.memory.add_user(question)
        self.memory.add_assistant(result.text)

        return Answer(
            text=result.text,
            citations=result.citations,
            tool_calls=[],
            confidence=result.confidence,
        )

    def ask_stream(self, question: str) -> Iterator[StreamChunk]:
        verdict = self.classifier.classify(question)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            yield StreamChunk(delta=verdict.refusal_message)
            return

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]
        retrieval = self.retriever.retrieve(
            queries=queries, top_k=self.top_k, candidate_k=self.candidate_k
        )
        full_text = ""
        for delta in self.synthesizer.synthesize_stream(
            question=question,
            retrieved=retrieval.chunks,
            canonical_chunks=[],
            history=self.memory.recent(n=_HISTORY_TURNS),
        ):
            full_text += delta
            yield StreamChunk(delta=delta)

        self.memory.add_user(question)
        self.memory.add_assistant(full_text)
