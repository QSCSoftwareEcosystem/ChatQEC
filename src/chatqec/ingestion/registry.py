"""Source registry: typed models, JSONL I/O, and lifecycle transitions.

The registry is the single source of truth for what is in the knowledge base.
It lives in Git (sources/registry.jsonl) and is human-auditable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SourceStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    STALE = "stale"
    PENDING_REVIEW = "pending_review"


class SourceType(str, Enum):
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
