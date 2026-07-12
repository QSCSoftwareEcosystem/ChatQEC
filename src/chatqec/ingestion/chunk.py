"""Chunk model + on-disk JSONL sink.

Chunks are the unit the embedder consumes. M4 produces them; M5 embeds them
and writes them to Qdrant.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from chatqec.ingestion.ids import safe_source_id


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    title: str
    source_type: str  # mirrors SourceType.value for embedding-time provenance
    authority_tier: str
    anchor: str = ""  # e.g. "#section-3", "?t=185s", "L42-L88"
    metadata: dict[str, object] = Field(default_factory=dict)


class ChunkSink:
    """Writes chunks for one source to data/chunks/<source_id>.jsonl atomically."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, source_id: str) -> str:
        return safe_source_id(source_id) + ".jsonl"

    def write_many(self, source_id: str, chunks: list[Chunk]) -> Path:
        out = self.root_dir / self._safe_filename(source_id)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json())
                f.write("\n")
        tmp.replace(out)
        return out
