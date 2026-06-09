"""Embedding pipeline: chunks on disk → embedder → Qdrant → registry promotion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.registry import (
    Registry,
    SourceStatus,
)
from chatqec.store.qdrant import ChatQecStore


class SparseEncoder(Protocol):
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]: ...


@dataclass(frozen=True)
class PipelineResult:
    source_id: str
    chunks_embedded: int


class EmbeddingPipeline:
    """Reads chunk JSONL files, embeds them, writes to Qdrant, flips registry status."""

    def __init__(
        self,
        *,
        chunks_dir: Path,
        registry: Registry,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        store: ChatQecStore,
    ) -> None:
        self.chunks_dir = Path(chunks_dir)
        self.registry = registry
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.store = store

    @staticmethod
    def _safe(source_id: str) -> str:
        return source_id.replace(":", "_").replace("/", "_").replace("@", "_")

    def embed_source(self, source_id: str) -> PipelineResult:
        path = self.chunks_dir / f"{self._safe(source_id)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        chunks = [
            Chunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not chunks:
            return PipelineResult(source_id=source_id, chunks_embedded=0)

        texts = [c.text for c in chunks]
        dense = self.embedder.embed_batch(texts)
        sparse = self.sparse_encoder.encode_batch(texts)

        self.store.upsert_chunks(chunks=chunks, dense_vectors=dense, sparse_vectors=sparse)

        # Promote registry status if currently pending_review
        rec = self.registry.get(source_id)
        if rec.status is SourceStatus.PENDING_REVIEW:
            self.registry.update_status(source_id, SourceStatus.ACTIVE)
        return PipelineResult(source_id=source_id, chunks_embedded=len(chunks))

    def embed_all_pending(self) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for rec in list(self.registry.filter(status=SourceStatus.PENDING_REVIEW)):
            results.append(self.embed_source(rec.id))
        return results

    def purge_retracted(
        self,
        *,
        older_than_days: int,
        now: datetime | None = None,
    ) -> list[str]:
        """Physically delete chunks from Qdrant for retracted/superseded sources."""
        now = now or datetime.now(tz=UTC)
        purged: list[str] = []
        for rec in self.registry.compact_candidates(
            older_than_days=older_than_days,
            now=now,
        ):
            self.store.delete_by_source_id(rec.id)
            purged.append(rec.id)
        return purged
