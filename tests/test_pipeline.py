"""End-to-end test: chunks.jsonl → embed → Qdrant → registry flipped to active."""

from datetime import datetime, UTC
from pathlib import Path
from collections.abc import Sequence

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk, ChunkSink
from chatqec.ingestion.pipeline import EmbeddingPipeline, PipelineResult
from chatqec.ingestion.registry import (
    Registry,
    SourceRecord,
    SourceStatus,
    SourceType,
)
from chatqec.store.qdrant import ChatQecStore, StoreConfig


class _StubEmbedder(Embedder):
    dim = 768

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        # Map first char to a 768-d ramp for deterministic dense vectors
        return [[float(ord(t[0])) / 256] * 768 for t in texts]


class _StubSparse:
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def _now() -> datetime:
    return datetime(2026, 5, 29, tzinfo=UTC)


def _seed(registry_path: Path, chunks_dir: Path, source_id: str = "arxiv:t1") -> None:
    sink = ChunkSink(chunks_dir)
    chunks = [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=f"chunk {i}",
            title="T",
            source_type="arxiv",
            authority_tier="preprint",
            anchor="",
        )
        for i in range(3)
    ]
    sink.write_many(source_id, chunks)
    reg = Registry(registry_path)
    reg.append(
        SourceRecord(
            id=source_id,
            type=SourceType.ARXIV,
            url=f"https://arxiv.org/abs/{source_id}",
            title="T",
            authority_tier="preprint",
            ingested_at=_now(),
            last_checked_at=_now(),
            status=SourceStatus.PENDING_REVIEW,
            chunk_ids=[c.chunk_id for c in chunks],
            chunk_count=len(chunks),
        )
    )


def test_pipeline_promotes_pending_to_active(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:t1")

    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir,
        registry=Registry(registry_path),
        embedder=_StubEmbedder(),
        sparse_encoder=_StubSparse(),
        store=store,
    )
    result = pipe.embed_source("arxiv:t1")
    assert isinstance(result, PipelineResult)
    assert result.chunks_embedded == 3
    assert store.count() == 3

    reg = Registry(registry_path)
    assert reg.get("arxiv:t1").status is SourceStatus.ACTIVE


def test_pipeline_embed_all_pending(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:a")
    _seed(registry_path, chunks_dir, "arxiv:b")

    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir,
        registry=Registry(registry_path),
        embedder=_StubEmbedder(),
        sparse_encoder=_StubSparse(),
        store=store,
    )
    results = pipe.embed_all_pending()
    assert sum(r.chunks_embedded for r in results) == 6
    assert store.count() == 6


def test_pipeline_compact_deletes_chunks_for_retracted_sources(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:keep")
    _seed(registry_path, chunks_dir, "arxiv:gone")

    store = ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768, on_disk_payload=False)
    )
    reg = Registry(registry_path)
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir,
        registry=reg,
        embedder=_StubEmbedder(),
        sparse_encoder=_StubSparse(),
        store=store,
    )
    pipe.embed_all_pending()
    assert store.count() == 6
    reg.update_status("arxiv:gone", SourceStatus.RETRACTED)

    purged = pipe.purge_retracted(older_than_days=0, now=datetime(2026, 12, 31, tzinfo=UTC))
    assert purged == ["arxiv:gone"]
    assert store.count() == 3
