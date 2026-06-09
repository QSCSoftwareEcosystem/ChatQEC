"""HybridRetriever end-to-end test using in-memory Qdrant + stub embedder/reranker."""

from datetime import date
from collections.abc import Sequence

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.hybrid import HybridRetriever, RetrievalResult
from chatqec.store.qdrant import ChatQecStore, StoreConfig


class _StubEmb:
    dim = 768

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(ord(t[0])) / 256] * 768 for t in texts]


class _StubSparse:
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


class _StubReranker:
    def __init__(self):
        self.calls: int = 0

    def rerank(self, query: str, candidates: list[str], *, top_k: int) -> list[int]:
        self.calls += 1
        # Identity rerank: return the candidates in original order
        return list(range(min(top_k, len(candidates))))


def _seed_store() -> ChatQecStore:
    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    chunks = [
        Chunk(
            chunk_id="a#0",
            source_id="a",
            chunk_index=0,
            text="surface code",
            title="Surface paper",
            source_type="arxiv",
            authority_tier="peer_reviewed",
            anchor="",
            metadata={"year": 2024},
        ),
        Chunk(
            chunk_id="b#0",
            source_id="b",
            chunk_index=0,
            text="color code",
            title="Color paper",
            source_type="arxiv",
            authority_tier="preprint",
            anchor="",
            metadata={"year": 2020},
        ),
    ]
    store.upsert_chunks(
        chunks=chunks,
        dense_vectors=[[0.5] * 768, [0.9] * 768],
        sparse_vectors=[
            SparseVector(indices=[1], values=[1.0]),
            SparseVector(indices=[2], values=[1.0]),
        ],
    )
    return store


def _canon_index_with_surface_code() -> CanonicalIndex:
    return CanonicalIndex(
        [
            CanonicalPage(
                frontmatter=CanonicalFrontmatter(
                    topic_slug="surface-code",
                    title="Surface Code",
                    aliases=["toric code"],
                    see_also=[],
                    authority_tier="canonical",
                    last_reviewed=date(2026, 5, 29),
                    maintainers=["m"],
                ),
                body="A canonical surface code intro.",
            )
        ]
    )


def test_retrieve_returns_topk_chunks():
    store = _seed_store()
    rr = HybridRetriever(
        store=store,
        embedder=_StubEmb(),
        sparse_encoder=_StubSparse(),
        reranker=_StubReranker(),
        canonical_index=_canon_index_with_surface_code(),
    )
    result = rr.retrieve(queries=["surface code"], top_k=2, candidate_k=10)
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) <= 2


def test_retrieve_force_injects_canonical_match():
    store = _seed_store()
    rr = HybridRetriever(
        store=store,
        embedder=_StubEmb(),
        sparse_encoder=_StubSparse(),
        reranker=_StubReranker(),
        canonical_index=_canon_index_with_surface_code(),
    )
    result = rr.retrieve(queries=["explain the toric code"], top_k=1, candidate_k=10)
    # "toric code" is an alias for surface-code → canonical is force-injected
    # Note: the canonical chunks aren't in Qdrant in this test (they would be
    # after M5 ingestion); the retriever still surfaces matched_topics.
    assert "surface-code" in result.matched_topics


def test_retrieve_reranker_called_once_per_query():
    store = _seed_store()
    reranker = _StubReranker()
    rr = HybridRetriever(
        store=store,
        embedder=_StubEmb(),
        sparse_encoder=_StubSparse(),
        reranker=reranker,
        canonical_index=_canon_index_with_surface_code(),
    )
    rr.retrieve(queries=["q1", "q2"], top_k=2, candidate_k=10)
    # Reranker is applied once over the union of all queries' candidates
    assert reranker.calls == 1


def test_retrieve_empty_store_returns_empty():
    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    rr = HybridRetriever(
        store=store,
        embedder=_StubEmb(),
        sparse_encoder=_StubSparse(),
        reranker=_StubReranker(),
        canonical_index=CanonicalIndex(pages=[]),
    )
    result = rr.retrieve(queries=["anything"], top_k=5, candidate_k=10)
    assert result.chunks == []
    assert result.matched_topics == []
