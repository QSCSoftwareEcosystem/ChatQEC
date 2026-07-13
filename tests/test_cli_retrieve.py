"""CLI test for `chatqec retrieve --query` debug command."""

from collections.abc import Sequence

from click.testing import CliRunner

from chatqec.cli import main
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.store.qdrant import ChatQecStore, StoreConfig


def test_retrieve_query_prints_table(monkeypatch):
    # Pre-populate a :memory: store and inject all the factory hooks
    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    store.upsert_chunks(
        chunks=[
            Chunk(
                chunk_id="x#0",
                source_id="x",
                chunk_index=0,
                text="surface code",
                title="S",
                source_type="arxiv",
                authority_tier="preprint",
                anchor="",
                metadata={"year": 2024},
            )
        ],
        dense_vectors=[[0.5] * 768],
        sparse_vectors=[SparseVector(indices=[1], values=[1.0])],
    )

    class _StubEmb:
        dim = 768

        def embed_batch(self, texts: Sequence[str]):
            return [[0.5] * 768 for _ in texts]

    class _StubSparse:
        def encode_batch(self, texts: Sequence[str]):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    class _StubReranker:
        def rerank(self, q, cands, *, top_k):
            return list(range(min(top_k, len(cands))))

        def rerank_scored(self, q, cands, *, top_k):
            idx = list(range(min(top_k, len(cands))))
            return [(i, 1.0 - 0.01 * pos) for pos, i in enumerate(idx)]

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", _StubSparse)
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll: store)
    monkeypatch.setattr("chatqec.cli._make_reranker", _StubReranker)
    monkeypatch.setattr(
        "chatqec.cli._make_canonical_index",
        lambda d: __import__(
            "chatqec.retrieval.canonical_injector", fromlist=["CanonicalIndex"]
        ).CanonicalIndex(pages=[]),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "retrieve",
            "--query",
            "surface code",
            "--top-k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "x#0" in result.output
