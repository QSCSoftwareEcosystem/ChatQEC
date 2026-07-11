"""ChatQEC public API (replaces the M1 stub).

End-to-end: classifier -> rewriter -> retriever -> synthesizer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from chatqec.api_types import Answer, Citation, ImageAttachment, StreamChunk, ToolCall
from chatqec.conversation.classifier import OffTopicClassifier
from chatqec.conversation.memory import ConversationMemory
from chatqec.conversation.rewriter import QueryRewriter
from chatqec.conversation.synthesizer import Synthesizer
from chatqec.conversation.vision import describe_image
from chatqec.ingestion.ids import canonical_wiki_id
from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.figure_loader import load_chunk_figures
from chatqec.retrieval.hybrid import HybridRetriever, RetrievalResult, RetrievedChunk
from chatqec.trace import trace

__all__ = [
    "Answer",
    "ChatQEC",
    "Citation",
    "StreamChunk",
    "ToolCall",
]

_HISTORY_TURNS = 6


def _chunk_figure_paths(
    chunks: list[RetrievedChunk], *, base_dir: Path = Path("."), max_figures: int = 3
) -> list[str]:
    """De-duplicated figure paths referenced by ``chunks``, capped at ``max_figures``.

    Mirrors the dedup/precedence order used by ``load_chunk_figures`` so the
    figures surfaced to the user match the ones the synthesizer could see.
    Only paths that exist under ``base_dir`` are included, so this list stays
    in sync with what ``load_chunk_figures`` actually loads.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        figs = c.payload.get("meta_figures") or c.payload.get("figures", [])
        if not isinstance(figs, list):
            continue
        for rel in figs:
            if rel in seen:
                continue
            seen.add(rel)
            if not (base_dir / rel).is_file():
                continue
            paths.append(rel)
            if len(paths) >= max_figures:
                return paths
    return paths


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

    def _canonical_chunks(self, matched_topics: list[str]) -> list[RetrievedChunk]:
        """Build force-injected canonical chunks for the matched topic slugs.

        Curated canonical pages bypass retrieval ranking: when a query mentions a
        canonical topic, its page is prepended to the synthesizer context so the
        answer can cite ``canonical:<slug>`` (spec §5.2/§8.4).
        """
        by_slug = {p.frontmatter.topic_slug: p for p in self.canonical_index.pages}
        chunks: list[RetrievedChunk] = []
        for slug in matched_topics:
            page = by_slug.get(slug)
            if page is None:
                continue
            source_id = canonical_wiki_id(slug)
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{source_id}#page",
                    source_id=source_id,
                    text=page.body.strip(),
                    title=page.frontmatter.title,
                    authority_tier="canonical",
                    score=1.0,
                    payload={"url": ""},
                )
            )
        return chunks

    def _caption(self, images: list[ImageAttachment]) -> str:
        """Describe attached images for retrieval (empty string if none)."""
        if not images:
            return ""
        return describe_image(client=self.rewriter.client, model=self.rewriter.model, images=images)

    def ask(self, question: str, images: list[ImageAttachment] | None = None) -> Answer:
        images = images or []
        trace("ask.question", question=question, top_k=self.top_k, candidate_k=self.candidate_k)
        caption = self._caption(images)
        trace("image.caption", n_images=len(images), caption=caption)
        classify_text = f"{question}\n{caption}".strip() if caption else question
        verdict = self.classifier.classify(classify_text)
        trace("classify.verdict", on_topic=verdict.on_topic, refusal=verdict.refusal_message)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            return Answer(text=verdict.refusal_message, confidence="low")

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]
        if caption:
            queries.append(caption)
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
        canonical_chunks = self._canonical_chunks(retrieval.matched_topics)
        trace(
            "canonical.injected",
            matched_topics=retrieval.matched_topics,
            injected=[c.source_id for c in canonical_chunks],
        )
        figure_images = images or load_chunk_figures(retrieval.chunks)
        figure_paths = [] if images else _chunk_figure_paths(retrieval.chunks)
        result = self.synthesizer.synthesize(
            question=question,
            retrieved=retrieval.chunks,
            canonical_chunks=canonical_chunks,
            history=self.memory.recent(n=_HISTORY_TURNS),
            images=figure_images,
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
            figure_paths=figure_paths,
        )

    def ask_stream(
        self, question: str, images: list[ImageAttachment] | None = None
    ) -> Iterator[StreamChunk]:
        images = images or []
        trace("ask_stream.question", question=question)
        caption = self._caption(images)
        trace("image.caption", n_images=len(images), caption=caption)
        classify_text = f"{question}\n{caption}".strip() if caption else question
        verdict = self.classifier.classify(classify_text)
        trace("classify.verdict", on_topic=verdict.on_topic, refusal=verdict.refusal_message)
        if not verdict.on_topic:
            self.memory.add_user(question)
            self.memory.add_assistant(verdict.refusal_message)
            yield StreamChunk(delta=verdict.refusal_message)
            return

        rewrite = self.rewriter.rewrite(question, memory=self.memory)
        queries = [rewrite.rewritten, *rewrite.variants]
        if caption:
            queries.append(caption)
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
        canonical_chunks = self._canonical_chunks(retrieval.matched_topics)
        trace(
            "canonical.injected",
            matched_topics=retrieval.matched_topics,
            injected=[c.source_id for c in canonical_chunks],
        )
        figure_images = images or load_chunk_figures(retrieval.chunks)
        full_text = ""
        for delta in self.synthesizer.synthesize_stream(
            question=question,
            retrieved=retrieval.chunks,
            canonical_chunks=canonical_chunks,
            history=self.memory.recent(n=_HISTORY_TURNS),
            images=figure_images,
        ):
            full_text += delta
            yield StreamChunk(delta=delta)

        self.memory.add_user(question)
        self.memory.add_assistant(full_text)
