"""CLI tests for `chatqec embed` — uses stub embedder via monkeypatch."""

from datetime import datetime, UTC
from pathlib import Path
from collections.abc import Sequence

from click.testing import CliRunner

from chatqec.cli import main
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk, ChunkSink
from chatqec.ingestion.registry import (
    Registry,
    SourceRecord,
    SourceStatus,
    SourceType,
)


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    reg_path = tmp_path / "reg.jsonl"
    sid = "arxiv:t1"
    sink = ChunkSink(chunks_dir)
    sink.write_many(
        sid,
        [
            Chunk(
                chunk_id=f"{sid}#0",
                source_id=sid,
                chunk_index=0,
                text="qec",
                title="T",
                source_type="arxiv",
                authority_tier="preprint",
                anchor="",
            )
        ],
    )
    Registry(reg_path).append(
        SourceRecord(
            id=sid,
            type=SourceType.ARXIV,
            url="https://arxiv.org/abs/t1",
            title="T",
            authority_tier="preprint",
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            last_checked_at=datetime(2026, 1, 1, tzinfo=UTC),
            status=SourceStatus.PENDING_REVIEW,
            chunk_ids=[f"{sid}#0"],
            chunk_count=1,
        )
    )
    return chunks_dir, reg_path


def _memory_store(coll: str):
    from chatqec.store.qdrant import ChatQecStore, StoreConfig

    return ChatQecStore(
        StoreConfig(qdrant_url=":memory:", collection=coll, vector_dim=768, on_disk_payload=False)
    )


def test_embed_source_flips_status_to_active(tmp_path: Path, monkeypatch):
    chunks_dir, reg_path = _seed(tmp_path)

    class _StubEmb:
        dim = 768

        def embed_batch(self, texts: Sequence[str]):
            return [[0.1] * 768 for _ in texts]

    class _StubSparse:
        def encode_batch(self, texts: Sequence[str]):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", _StubSparse)
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll: _memory_store(coll))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--chunks-dir",
            str(chunks_dir),
            "--registry",
            str(reg_path),
            "--source-id",
            "arxiv:t1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 chunks" in result.output

    reg = Registry(reg_path)
    assert reg.get("arxiv:t1").status is SourceStatus.ACTIVE


def test_embed_all_pending_processes_multiple(tmp_path: Path, monkeypatch):
    chunks_dir, reg_path = _seed(tmp_path)
    # Add a second pending source
    sid2 = "arxiv:t2"
    ChunkSink(chunks_dir).write_many(
        sid2,
        [
            Chunk(
                chunk_id=f"{sid2}#0",
                source_id=sid2,
                chunk_index=0,
                text="x",
                title="T2",
                source_type="arxiv",
                authority_tier="preprint",
                anchor="",
            )
        ],
    )
    Registry(reg_path).append(
        SourceRecord(
            id=sid2,
            type=SourceType.ARXIV,
            url="https://arxiv.org/abs/t2",
            title="T2",
            authority_tier="preprint",
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            last_checked_at=datetime(2026, 1, 1, tzinfo=UTC),
            status=SourceStatus.PENDING_REVIEW,
            chunk_ids=[f"{sid2}#0"],
            chunk_count=1,
        )
    )

    class _StubEmb:
        dim = 768

        def embed_batch(self, texts):
            return [[0.1] * 768 for _ in texts]

    class _StubSparse:
        def encode_batch(self, texts):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", _StubSparse)
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll: _memory_store(coll))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--chunks-dir",
            str(chunks_dir),
            "--registry",
            str(reg_path),
            "--all-pending",
        ],
    )
    assert result.exit_code == 0
    assert "2 sources" in result.output
