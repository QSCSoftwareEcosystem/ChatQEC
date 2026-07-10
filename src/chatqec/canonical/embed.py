"""Stage canonical wiki pages into the embed pipeline.

Canonical pages are force-injected from disk at query time, but to also
participate in semantic (hybrid) retrieval they must be chunked, written to the
chunk store, and registered as ``CANONICAL_WIKI`` sources. ``chatqec embed`` then
vectorizes them into Qdrant exactly like any other source.

This module produces the on-disk artifacts (chunk JSONL + registry records); it
performs no embedding itself and has no external dependencies, so it is fully
unit-testable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from chatqec.canonical.chunker import chunk_canonical_body
from chatqec.canonical.loader import load_canonical_directory
from chatqec.canonical.page import CanonicalPage
from chatqec.ingestion.chunk import Chunk, ChunkSink
from chatqec.ingestion.ids import canonical_wiki_id
from chatqec.ingestion.registry import (
    Registry,
    RegistryError,
    SourceRecord,
    SourceStatus,
    SourceType,
)

_SOURCE_TYPE = SourceType.CANONICAL_WIKI.value  # "canonical_wiki"


def build_canonical_chunks(page: CanonicalPage) -> list[Chunk]:
    """Chunk a canonical page into embeddable ``Chunk`` records.

    One chunk per ``##`` section (plus a leading preamble chunk if the page has
    body text before its first heading), mirroring ``chunk_canonical_body``.
    """
    source_id = canonical_wiki_id(page.frontmatter.topic_slug)
    title = page.frontmatter.title
    sections = chunk_canonical_body(page.body)
    return [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=section.text,
            title=title,
            source_type=_SOURCE_TYPE,
            authority_tier="canonical",
            anchor=f"#{section.anchor}" if section.anchor else "",
            metadata={"section": section.section_title},
        )
        for i, section in enumerate(sections)
    ]


def build_canonical_record(
    page: CanonicalPage,
    chunks: list[Chunk],
    *,
    now: datetime,
) -> SourceRecord:
    """Build the ``pending_review`` registry record for a canonical page."""
    source_id = canonical_wiki_id(page.frontmatter.topic_slug)
    return SourceRecord(
        id=source_id,
        type=SourceType.CANONICAL_WIKI,
        url="",
        title=page.frontmatter.title,
        authors=list(page.frontmatter.maintainers),
        authority_tier="canonical",
        ingested_at=now,
        last_checked_at=now,
        status=SourceStatus.PENDING_REVIEW,
        chunk_count=len(chunks),
        chunk_ids=[c.chunk_id for c in chunks],
        verified=True,
        notes=f"canonical wiki page (last_reviewed={page.frontmatter.last_reviewed.isoformat()})",
    )


def stage_canonical_directory(
    directory: Path,
    *,
    chunks_dir: Path,
    registry: Registry,
    now: datetime,
) -> list[str]:
    """Chunk every canonical page, write chunk files, and register sources.

    Returns the staged ``canonical:<slug>`` source ids. Idempotent: chunk files
    are overwritten atomically and an already-registered source is left in place
    (its status is not reset), so re-running after editing pages is safe.
    """
    sink = ChunkSink(chunks_dir)
    staged: list[str] = []
    for page in load_canonical_directory(directory):
        chunks = build_canonical_chunks(page)
        if not chunks:
            continue
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        try:
            registry.get(source_id)
        except RegistryError:
            registry.append(build_canonical_record(page, chunks, now=now))
        staged.append(source_id)
    return staged
