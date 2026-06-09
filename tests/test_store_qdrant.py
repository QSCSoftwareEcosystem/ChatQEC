"""Qdrant store wrapper tests using the in-memory client."""

from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.store.qdrant import ChatQecStore, StoreConfig


def _chunk(i: int, source_id: str = "test:1", text: str = "qec") -> Chunk:
    return Chunk(
        chunk_id=f"{source_id}#{i}",
        source_id=source_id,
        chunk_index=i,
        text=text,
        title="T",
        source_type="arxiv",
        authority_tier="preprint",
        anchor="",
    )


def _dense(seed: float = 0.1) -> list[float]:
    return [seed] * 768


def _sparse() -> SparseVector:
    return SparseVector(indices=[1, 5, 9], values=[0.7, 0.3, 0.1])


def test_store_creates_collection_on_init():
    cfg = StoreConfig(
        qdrant_url=":memory:", collection="test", vector_dim=768, on_disk_payload=False
    )
    store = ChatQecStore(cfg)
    assert store.collection_exists()


def test_store_upserts_and_retrieves_by_id():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0), _chunk(1)],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    assert store.count() == 2


def _axis(i: int) -> list[float]:
    """A unit vector pointing along axis ``i`` (directionally distinct so cosine
    distance can actually separate candidates)."""
    v = [0.0] * 768
    v[i] = 1.0
    return v


def test_store_dense_search_returns_nearest():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0, text="surface"), _chunk(1, text="color")],
        dense_vectors=[_axis(0), _axis(1)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    # Query points along axis 0 → chunk #0 (cosine 1.0) beats chunk #1 (cosine 0.0).
    hits = store.dense_search(query_vector=_axis(0), top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "test:1#0"


def test_store_filter_excludes_retracted():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0), _chunk(1)],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    store.set_status_for_source("test:1", "retracted")
    hits = store.dense_search(query_vector=_dense(0.1), top_k=10, status_filter="active")
    assert hits == []


def test_store_delete_by_source_id():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0, source_id="a"), _chunk(1, source_id="a"), _chunk(0, source_id="b")],
        dense_vectors=[_dense(0.1), _dense(0.2), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse(), _sparse()],
    )
    store.delete_by_source_id("a")
    assert store.count() == 1


def test_sparse_search_returns_results_for_overlapping_terms():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0, text="surface"), _chunk(1, text="color")],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[
            SparseVector(indices=[10, 20], values=[1.0, 0.5]),
            SparseVector(indices=[30, 40], values=[1.0, 0.5]),
        ],
    )
    # Query with sparse vector that overlaps with the first chunk's indices
    hits = store.sparse_search(
        sparse_query=SparseVector(indices=[10], values=[1.0]),
        top_k=10,
    )
    assert len(hits) >= 1
    # First chunk should rank higher because its sparse vector contains index 10
    assert hits[0].chunk_id == "test:1#0"


def test_sparse_search_filters_by_status():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0)],
        dense_vectors=[_dense(0.1)],
        sparse_vectors=[SparseVector(indices=[1], values=[1.0])],
    )
    store.set_status_for_source("test:1", "retracted")
    hits = store.sparse_search(
        sparse_query=SparseVector(indices=[1], values=[1.0]),
        top_k=10,
        status_filter="active",
    )
    assert hits == []
