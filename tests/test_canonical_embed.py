"""Staging canonical wiki pages into the embed pipeline (chunks + registry).

Canonical pages are force-injected from disk, but to also participate in
semantic (hybrid) retrieval they must be chunked, written to data/chunks, and
registered as CANONICAL_WIKI sources so `chatqec embed` can vectorize them.
"""

from datetime import UTC, datetime

from chatqec.canonical.embed import (
    build_canonical_chunks,
    build_canonical_record,
    stage_canonical_directory,
)
from chatqec.canonical.page import parse_page
from chatqec.ingestion.registry import Registry, SourceStatus, SourceType

NOW = datetime(2026, 7, 10, tzinfo=UTC)

PAGE_TEXT = """---
topic_slug: sample-code
title: "Sample Code"
aliases: ["sample"]
see_also: []
authority_tier: canonical
last_reviewed: 2026-07-10
maintainers: ["sharmin"]
---

# Sample Code

Preamble paragraph.

## First Section

Body of the first section.

## Second Section

Body of the second section.
"""


def _page():
    return parse_page(PAGE_TEXT)


def test_build_canonical_chunks_one_per_section_plus_preamble():
    chunks = build_canonical_chunks(_page())
    # preamble + 2 H2 sections
    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_build_canonical_chunks_carry_canonical_provenance():
    chunks = build_canonical_chunks(_page())
    for c in chunks:
        assert c.source_id == "canonical:sample-code"
        assert c.source_type == "canonical_wiki"
        assert c.authority_tier == "canonical"
        assert c.title == "Sample Code"


def test_build_canonical_chunks_id_format_and_anchors():
    chunks = build_canonical_chunks(_page())
    assert [c.chunk_id for c in chunks] == [
        "canonical:sample-code#0",
        "canonical:sample-code#1",
        "canonical:sample-code#2",
    ]
    # preamble has no anchor; sections get a kebab anchor of their heading
    assert chunks[0].anchor == ""
    assert chunks[1].anchor == "#first-section"
    assert chunks[2].anchor == "#second-section"


def test_build_canonical_record_is_pending_with_provenance():
    page = _page()
    chunks = build_canonical_chunks(page)
    rec = build_canonical_record(page, chunks, now=NOW)
    assert rec.id == "canonical:sample-code"
    assert rec.type is SourceType.CANONICAL_WIKI
    assert rec.authority_tier == "canonical"
    assert rec.status is SourceStatus.PENDING_REVIEW
    assert rec.chunk_count == 3
    assert rec.chunk_ids == [c.chunk_id for c in chunks]
    assert rec.authors == ["sharmin"]
    assert rec.ingested_at == NOW


def test_stage_directory_writes_chunk_files_and_registers(tmp_path):
    canon_dir = tmp_path / "canonical"
    canon_dir.mkdir()
    (canon_dir / "sample-code.md").write_text(PAGE_TEXT, encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    registry = Registry(tmp_path / "registry.jsonl")

    staged = stage_canonical_directory(canon_dir, chunks_dir=chunks_dir, registry=registry, now=NOW)

    assert staged == ["canonical:sample-code"]
    chunk_file = chunks_dir / "canonical_sample-code.jsonl"
    assert chunk_file.exists()
    assert len([ln for ln in chunk_file.read_text().splitlines() if ln.strip()]) == 3
    rec = registry.get("canonical:sample-code")
    assert rec.status is SourceStatus.PENDING_REVIEW
    assert rec.type is SourceType.CANONICAL_WIKI


def test_stage_directory_is_idempotent(tmp_path):
    canon_dir = tmp_path / "canonical"
    canon_dir.mkdir()
    (canon_dir / "sample-code.md").write_text(PAGE_TEXT, encoding="utf-8")
    chunks_dir = tmp_path / "chunks"
    registry = Registry(tmp_path / "registry.jsonl")

    stage_canonical_directory(canon_dir, chunks_dir=chunks_dir, registry=registry, now=NOW)
    # Second run must not raise (duplicate id) and must not duplicate records.
    staged2 = stage_canonical_directory(
        canon_dir, chunks_dir=chunks_dir, registry=registry, now=NOW
    )

    assert staged2 == ["canonical:sample-code"]
    assert len([r for r in registry.all() if r.id == "canonical:sample-code"]) == 1
