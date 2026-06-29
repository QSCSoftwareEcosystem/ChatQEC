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
from chatqec.retrieval.hybrid import HybridRetriever, RetrievalResult
from chatqec.trace import trace

__all__ = [
    "Answer",
    "ChatQEC",
    "Citation",
    "StreamChunk",
    "ToolCall",
]

_HISTORY_TURNS = 6


def _trace_retrieval(retrieval: RetrievalResult) -> None:
    """Emit a per-chunk trace of what retrieval surfaced (full chunk text)."""
    trace(
        "retrieval.result",
        matched_topics=retrieval.matched_topics,
        n_chunks=len(retrieval.chunks),
        debug=retrieval.debug,
    )
    for i, c in enumerate(retrieval.chunks):
        trace(
            f"retrieval.chunk[{i}]",
            chunk_id=c.chunk_id,
            source_id=c.source_id,
            title=c.title,
            authority_tier=c.authority_tier,
            score=round(c.score, 4),
            text=c.text,
        )


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
        trace("ask.question", question=question, top_k=self.top_k, candidate_k=self.candidate_k)
        verdict = self.classifier.classify(question)
        trace("classify.verdict", on_topic=verdict.on_topic, refusal=verdict.refusal_message)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            return Answer(text=verdict.refusal_message, confidence="low")

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]
        trace(
            "rewrite.result",
            rewritten=rewrite.rewritten,
            variants=rewrite.variants,
            is_followup=rewrite.is_followup,
            queries=queries,
        )

        retrieval = self.retriever.retrieve(
            queries=queries, top_k=self.top_k, candidate_k=self.candidate_k
        )
        _trace_retrieval(retrieval)
        result = self.synthesizer.synthesize(
            question=question,
            retrieved=retrieval.chunks,
            canonical_chunks=[],
            history=self.memory.recent(n=_HISTORY_TURNS),
        )
        trace(
            "synthesize.result",
            confidence=result.confidence,
            n_citations=len(result.citations),
            citations=[(c.source_id, c.title) for c in result.citations],
            n_tool_calls=len(result.tool_invocations),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        self.memory.add_user(question)
        self.memory.add_assistant(result.text)

        return Answer(
            text=result.text,
            citations=result.citations,
            tool_calls=[
                ToolCall(name=t.name, input=dict(t.input), output=str(t.output))
                for t in result.tool_invocations
            ],
            confidence=result.confidence,
        )

    def ask_stream(self, question: str) -> Iterator[StreamChunk]:
        trace("ask_stream.question", question=question)
        verdict = self.classifier.classify(question)
        trace("classify.verdict", on_topic=verdict.on_topic, refusal=verdict.refusal_message)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            yield StreamChunk(delta=verdict.refusal_message)
            return

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]
        trace(
            "rewrite.result",
            rewritten=rewrite.rewritten,
            variants=rewrite.variants,
            is_followup=rewrite.is_followup,
            queries=queries,
        )
        retrieval = self.retriever.retrieve(
            queries=queries, top_k=self.top_k, candidate_k=self.candidate_k
        )
        _trace_retrieval(retrieval)
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
