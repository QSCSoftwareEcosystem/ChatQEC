"""Source registry: typed models, JSONL I/O, and lifecycle transitions.

The registry is the single source of truth for what is in the knowledge base.
It lives in Git (sources/registry.jsonl) and is human-auditable.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SourceStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    STALE = "stale"
    PENDING_REVIEW = "pending_review"


class SourceType(StrEnum):
    ARXIV = "arxiv"
    GITHUB = "github"
    YOUTUBE = "youtube"
    BLOG = "blog"
    PDF = "pdf"
    CANONICAL_WIKI = "canonical_wiki"


class SourceRecord(BaseModel):
    """One row in sources/registry.jsonl. Immutable once written.

    Lifecycle fields (status, supersedes, superseded_by) are mutated only via
    Registry.update_status() so transitions are validated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: SourceType
    url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    version: str | None = None
    content_sha256: str | None = None
    license: str | None = None
    authority_tier: str  # canonical | peer_reviewed | preprint | blog | video
    ingested_at: datetime
    last_checked_at: datetime
    status: SourceStatus
    supersedes: str | None = None
    superseded_by: str | None = None
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    verified: bool = False
    notes: str | None = None


# --- Registry I/O and lifecycle ---


class RegistryError(Exception):
    """Base class for registry errors."""


class RecordNotFoundError(RegistryError):
    """Raised when a record id is not present in the registry."""


class InvalidTransitionError(RegistryError):
    """Raised when an attempted status transition is not permitted."""


# Allowed lifecycle transitions. See spec §6.
_ALLOWED_TRANSITIONS: dict[SourceStatus, set[SourceStatus]] = {
    SourceStatus.PENDING_REVIEW: {SourceStatus.ACTIVE, SourceStatus.RETRACTED},
    SourceStatus.ACTIVE: {SourceStatus.SUPERSEDED, SourceStatus.RETRACTED, SourceStatus.STALE},
    SourceStatus.STALE: {SourceStatus.ACTIVE, SourceStatus.SUPERSEDED, SourceStatus.RETRACTED},
    SourceStatus.SUPERSEDED: {SourceStatus.RETRACTED},
    SourceStatus.RETRACTED: set(),
}


class Registry:
    """JSONL-backed source registry.

    Records are append-only on disk; updates rewrite the file atomically.
    For v1 corpora (≤ 10k records) the in-memory representation is fine.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._records: dict[str, SourceRecord] = self._load()

    def _load(self) -> dict[str, SourceRecord]:
        records: dict[str, SourceRecord] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                rec = SourceRecord.model_validate_json(stripped)
                records[rec.id] = rec
        return records

    def _flush(self) -> None:
        """Rewrite the JSONL file atomically."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(rec.model_dump_json())
                f.write("\n")
        tmp.replace(self.path)

    def append(self, record: SourceRecord) -> None:
        if record.id in self._records:
            raise RegistryError(f"duplicate id: {record.id}")
        self._records[record.id] = record
        self._flush()

    def get(self, record_id: str) -> SourceRecord:
        try:
            return self._records[record_id]
        except KeyError as e:
            raise RecordNotFoundError(record_id) from e

    def all(self) -> Iterator[SourceRecord]:
        return iter(self._records.values())

    def filter(self, *, status: SourceStatus | None = None) -> Iterator[SourceRecord]:
        for rec in self._records.values():
            if status is not None and rec.status is not status:
                continue
            yield rec

    def update_status(self, record_id: str, new_status: SourceStatus) -> SourceRecord:
        rec = self.get(record_id)
        if new_status not in _ALLOWED_TRANSITIONS[rec.status]:
            raise InvalidTransitionError(
                f"cannot move {record_id} from {rec.status.value} → {new_status.value}"
            )
        updated = rec.model_copy(update={"status": new_status})
        self._records[record_id] = updated
        self._flush()
        return updated

    def supersede(self, *, old_id: str, new_id: str) -> tuple[SourceRecord, SourceRecord]:
        old = self.get(old_id)
        new = self.get(new_id)
        if old.status is not SourceStatus.ACTIVE:
            raise InvalidTransitionError(
                f"can only supersede an ACTIVE record; {old_id} is {old.status.value}"
            )
        if new.status in (SourceStatus.RETRACTED, SourceStatus.SUPERSEDED):
            raise InvalidTransitionError(
                f"supersede target {new_id} cannot be {new.status.value}; "
                f"must be ACTIVE, PENDING_REVIEW, or STALE"
            )
        old = old.model_copy(
            update={
                "status": SourceStatus.SUPERSEDED,
                "superseded_by": new_id,
            }
        )
        new = new.model_copy(update={"supersedes": old_id})
        self._records[old_id] = old
        self._records[new_id] = new
        self._flush()
        return old, new

    def compact_candidates(
        self,
        *,
        older_than_days: int,
        now: datetime,
    ) -> Iterator[SourceRecord]:
        cutoff = now - timedelta(days=older_than_days)
        for rec in self._records.values():
            if (
                rec.status in (SourceStatus.RETRACTED, SourceStatus.SUPERSEDED)
                and rec.last_checked_at < cutoff
            ):
                yield rec
