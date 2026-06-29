"""HybridRetriever — orchestrates dense + BM25 search, scoring, reranking, and
canonical force-injection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import SparseVector
from chatqec.retrieval.canonical_injector import CanonicalIndex, match_topics
from chatqec.retrieval.normalize import identity_if_single
from chatqec.retrieval.reranker import CrossEncoderReranker
from chatqec.retrieval.scoring import (
    Candidate,
    Coefficients,
    score_candidates,
)
from chatqec.store.qdrant import ChatQecStore
from chatqec.trace import trace


class SparseEncoderProto(Protocol):
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    text: str
    title: str
    authority_tier: str
    score: float
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    matched_topics: list[str]
    debug: dict[str, object] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        *,
        store: ChatQecStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoderProto,
        reranker: CrossEncoderReranker,
        canonical_index: CanonicalIndex,
        coefficients: Coefficients | None = None,
        per_query_k: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.canonical_index = canonical_index
        self.coefficients = coefficients or Coefficients()
        self.per_query_k = per_query_k

    def retrieve(
        self,
        *,
        queries: list[str],
        top_k: int,
        candidate_k: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        now = now or datetime.now(tz=UTC)
        trace(
            "retriever.retrieve",
            queries=queries,
            top_k=top_k,
            candidate_k=candidate_k,
            per_query_k=self.per_query_k,
        )
        if not queries:
            return RetrievalResult(chunks=[], matched_topics=[])

        # 1) embed queries
        dense_q = self.embedder.embed_batch(queries)
        sparse_q = self.sparse_encoder.encode_batch(queries)
        trace("retriever.embedded", n_dense=len(dense_q), n_sparse=len(sparse_q))

        # 2) per-query dense + sparse search → unioned hit map
        hits_by_chunk: dict[str, dict[str, object]] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for dv in dense_q:
            for h in self.store.dense_search(query_vector=dv, top_k=self.per_query_k):
                hits_by_chunk[h.chunk_id] = h.payload
                dense_scores[h.chunk_id] = max(dense_scores.get(h.chunk_id, 0.0), h.score)
        for sv in sparse_q:
            for h in self.store.sparse_search(sparse_query=sv, top_k=self.per_query_k):
                hits_by_chunk[h.chunk_id] = h.payload
                bm25_scores[h.chunk_id] = max(bm25_scores.get(h.chunk_id, 0.0), h.score)

        trace(
            "retriever.searched",
            dense_hits=len(dense_scores),
            bm25_hits=len(bm25_scores),
            unique_chunks=len(hits_by_chunk),
        )
        if not hits_by_chunk:
            matched_topics = match_topics(" ".join(queries), self.canonical_index)
            trace("retriever.no_hits", matched_topics=matched_topics)
            return RetrievalResult(chunks=[], matched_topics=matched_topics)

        # 3) normalize and score per §8.5
        chunk_ids = list(hits_by_chunk.keys())
        dn = identity_if_single([dense_scores.get(c, 0.0) for c in chunk_ids])
        bn = identity_if_single([bm25_scores.get(c, 0.0) for c in chunk_ids])
        candidates: list[Candidate] = []
        for cid, dnorm, bnorm in zip(chunk_ids, dn, bn, strict=True):
            payload = hits_by_chunk[cid]
            year_raw = payload.get("meta_year") or payload.get("year")
            year = int(year_raw) if isinstance(year_raw, int | float) else None
            candidates.append(
                Candidate(
                    chunk_id=cid,
                    dense_norm=dnorm,
                    bm25_norm=bnorm,
                    authority_tier=str(payload.get("authority_tier", "blog")),
                    year=year,
                )
            )
        scored = score_candidates(candidates, coefficients=self.coefficients, now=now)

        # 4) take top candidate_k
        top_cands = scored[:candidate_k]
        trace(
            "retriever.scored",
            n_candidates=len(candidates),
            n_after_topk=len(top_cands),
            top_scored=[(s.chunk_id, round(s.final_score, 4)) for s in top_cands[:top_k]],
        )

        # 5) rerank → top_k (called ONCE over the union of all queries' candidates)
        texts = [str(hits_by_chunk[s.chunk_id].get("text", "")) for s in top_cands]
        # Use first query for reranking (caller can pass merged query)
        rerank_query = queries[0]
        top_idx = self.reranker.rerank(rerank_query, texts, top_k=top_k)
        trace("retriever.reranked", rerank_query=rerank_query, top_idx=list(top_idx))
        retrieved = [
            RetrievedChunk(
                chunk_id=top_cands[i].chunk_id,
                source_id=str(hits_by_chunk[top_cands[i].chunk_id].get("source_id", "")),
                text=str(hits_by_chunk[top_cands[i].chunk_id].get("text", "")),
                title=str(hits_by_chunk[top_cands[i].chunk_id].get("title", "")),
                authority_tier=str(hits_by_chunk[top_cands[i].chunk_id].get("authority_tier", "")),
                score=top_cands[i].final_score,
                payload=hits_by_chunk[top_cands[i].chunk_id],
            )
            for i in top_idx
        ]

        # 6) canonical topic match (force-injection signaled to caller)
        matched_topics = match_topics(" ".join(queries), self.canonical_index)

        return RetrievalResult(
            chunks=retrieved,
            matched_topics=matched_topics,
            debug={
                "candidates_considered": len(candidates),
                "after_scoring": len(top_cands),
                "after_rerank": len(retrieved),
            },
        )
