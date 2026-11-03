"""ChatQEC public API (replaces the M1 stub).

End-to-end: classifier -> rewriter -> retriever -> synthesizer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
from chatqec.retrieval.web_search import WebResult, build_web_chunks
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
        web_searcher: Callable[[str, int], list[WebResult]] | None = None,
        web_fallback: bool = False,
        # Absolute mxbai-rerank cross-encoder score of the best chunk (NOT the
        # min-max-normalized retrieval score). Calibrated 2026-07-13 against real
        # queries: covered/in-corpus topics score 0.52-0.99, genuinely-uncovered QEC
        # specifics 0.15-0.33, off-topic <0.03. 0.4 sits in the gap — fires the web
        # fallback for uncovered/off-topic queries, never for covered topics.
        web_fallback_threshold: float = 0.4,
        web_k: int = 5,
        wiki_searcher: Callable[[str, int], list[RetrievedChunk]] | None = None,
        wiki_k: int = 8,
    ) -> None:
        self.classifier = classifier
        self.rewriter = rewriter
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.canonical_index = canonical_index
        self.memory = memory or ConversationMemory()
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.web_searcher = web_searcher
        self.web_fallback = web_fallback
        self.web_fallback_threshold = web_fallback_threshold
        self.web_k = web_k
        self.wiki_searcher = wiki_searcher
        self.wiki_k = wiki_k

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

    def _web_chunks(self, question: str, retrieval: RetrievalResult) -> list[RetrievedChunk]:
        """Gate-guarded web-search fallback chunks (empty unless enabled + weak retrieval)."""
        searcher = self.web_searcher if self.web_fallback else None
        return build_web_chunks(
            question,
            retrieval=retrieval,
            web_searcher=searcher,
            threshold=self.web_fallback_threshold,
            web_k=self.web_k,
        )

    def _wiki_chunks(self, query: str) -> list[RetrievedChunk]:
        """Wiki sections for the rerank pool. Best-effort: never breaks an answer.

        Takes the *rewritten* query, not the raw question. The wiki serves a lexical
        BM25 index, which is exactly the retriever that cannot recover a topic the
        user left implicit: on "what about its threshold?" the raw question carries
        no topic terms at all, while the rewrite has resolved the pronoun. It also
        matches what the reranker scores against (``queries[0]``).
        """
        if self.wiki_searcher is None:
            return []
        try:
            return self.wiki_searcher(query, self.wiki_k)
        except Exception:  # the wiki is a bonus source, never a dependency
            trace("wiki.search_failed", query=query)
            return []

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
        verdict = self.classifier.classify(classify_text, memory=self.memory)
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

        wiki_extra = self._wiki_chunks(queries[0])
        retrieval = self.retriever.retrieve(
            queries=queries,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            extra_chunks=wiki_extra,
        )
        _trace_retrieval(retrieval)
        canonical_chunks = self._canonical_chunks(retrieval.matched_topics)
        trace(
            "canonical.injected",
            matched_topics=retrieval.matched_topics,
            injected=[c.source_id for c in canonical_chunks],
        )
        web_chunks = self._web_chunks(question, retrieval)
        # Skip generic corpus figures when MCP tools are available: a tool-driven
        # answer renders its own circuit/plot, and unrelated textbook figures only
        # add noise (they'd also cost vision tokens for no benefit — see
        # docs/stim_getting_started_capability_check.md).
        mcp_enabled = self.synthesizer.mcp_adapter is not None
        figure_images = images or ([] if mcp_enabled else load_chunk_figures(retrieval.chunks))
        figure_paths = [] if images or mcp_enabled else _chunk_figure_paths(retrieval.chunks)
        result = self.synthesizer.synthesize(
            question=question,
            retrieved=retrieval.chunks + web_chunks,
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
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def ask_stream(
        self, question: str, images: list[ImageAttachment] | None = None
    ) -> Iterator[StreamChunk]:
        images = images or []
        trace("ask_stream.question", question=question)
        caption = self._caption(images)
        trace("image.caption", n_images=len(images), caption=caption)
        classify_text = f"{question}\n{caption}".strip() if caption else question
        verdict = self.classifier.classify(classify_text, memory=self.memory)
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
        wiki_extra = self._wiki_chunks(queries[0])
        retrieval = self.retriever.retrieve(
            queries=queries,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            extra_chunks=wiki_extra,
        )
        _trace_retrieval(retrieval)
        canonical_chunks = self._canonical_chunks(retrieval.matched_topics)
        trace(
            "canonical.injected",
            matched_topics=retrieval.matched_topics,
            injected=[c.source_id for c in canonical_chunks],
        )
        web_chunks = self._web_chunks(question, retrieval)
        mcp_enabled = self.synthesizer.mcp_adapter is not None
        figure_images = images or ([] if mcp_enabled else load_chunk_figures(retrieval.chunks))
        full_text = ""
        for delta in self.synthesizer.synthesize_stream(
            question=question,
            retrieved=retrieval.chunks + web_chunks,
            canonical_chunks=canonical_chunks,
            history=self.memory.recent(n=_HISTORY_TURNS),
            images=figure_images,
        ):
            full_text += delta
            yield StreamChunk(delta=delta)

        self.memory.add_user(question)
        self.memory.add_assistant(full_text)
