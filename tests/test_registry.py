"""Source registry model + I/O tests."""

import json
from datetime import datetime, UTC
from pathlib import Path

import pytest
from pydantic import ValidationError

from chatqec.ingestion.registry import (
    InvalidTransitionError,
    RecordNotFoundError,
    Registry,
    SourceRecord,
    SourceStatus,
    SourceType,
)


def _utc(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_source_status_has_five_v1_states():
    expected = {"active", "superseded", "retracted", "stale", "pending_review"}
    assert {s.value for s in SourceStatus} == expected


def test_source_type_has_v1_types():
    expected = {"arxiv", "github", "youtube", "blog", "pdf", "canonical_wiki"}
    assert {t.value for t in SourceType} == expected


def test_source_record_minimum_fields():
    rec = SourceRecord(
        id="arxiv:2408.13687v1",
        type=SourceType.ARXIV,
        url="https://arxiv.org/abs/2408.13687",
        title="Quantum error correction below the surface code threshold",
        authority_tier="peer_reviewed",
        ingested_at=_utc(2026, 5, 29),
        last_checked_at=_utc(2026, 5, 29),
        status=SourceStatus.ACTIVE,
    )
    assert rec.id == "arxiv:2408.13687v1"
    assert rec.chunk_count == 0
    assert rec.chunk_ids == []
    assert rec.verified is False  # default


def test_source_record_rejects_unknown_status():
    with pytest.raises(ValidationError):
        SourceRecord(
            id="x",
            type=SourceType.ARXIV,
            url="https://example.com",
            title="t",
            authority_tier="preprint",
            ingested_at=_utc(2026, 1, 1),
            last_checked_at=_utc(2026, 1, 1),
            status="not_a_status",  # type: ignore[arg-type]
        )


def test_source_record_round_trips_through_json():
    rec = SourceRecord(
        id="arxiv:2408.13687v1",
        type=SourceType.ARXIV,
        url="https://arxiv.org/abs/2408.13687",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        version="v1",
        content_sha256="ab12",
        authority_tier="peer_reviewed",
        ingested_at=_utc(2026, 5, 29),
        last_checked_at=_utc(2026, 5, 29),
        status=SourceStatus.ACTIVE,
        verified=True,
    )
    blob = rec.model_dump_json()
    decoded = SourceRecord.model_validate_json(blob)
    assert decoded == rec
    # JSON shape sanity
    parsed = json.loads(blob)
    assert parsed["status"] == "active"
    assert parsed["type"] == "arxiv"


# --- Registry I/O and lifecycle tests ---


def _make(rid: str, status: SourceStatus = SourceStatus.ACTIVE) -> SourceRecord:
    return SourceRecord(
        id=rid,
        type=SourceType.ARXIV,
        url=f"https://arxiv.org/abs/{rid}",
        title=f"Title {rid}",
        authority_tier="preprint",
        ingested_at=_utc(2026, 1, 1),
        last_checked_at=_utc(2026, 1, 1),
        status=status,
    )


def test_registry_append_and_read_round_trip(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    reg.append(_make("arxiv:1111.11111v1"))
    reg.append(_make("arxiv:2222.22222v1", status=SourceStatus.PENDING_REVIEW))

    reg2 = Registry(path)
    all_records = list(reg2.all())
    assert len(all_records) == 2
    assert {r.id for r in all_records} == {"arxiv:1111.11111v1", "arxiv:2222.22222v1"}


def test_registry_get_by_id(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    reg.append(_make("arxiv:1111.11111v1"))
    rec = reg.get("arxiv:1111.11111v1")
    assert rec.title == "Title arxiv:1111.11111v1"


def test_registry_get_missing_raises(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    with pytest.raises(RecordNotFoundError):
        reg.get("arxiv:nope")


def test_registry_filter_by_status(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.ACTIVE))
    reg.append(_make("b", status=SourceStatus.RETRACTED))
    reg.append(_make("c", status=SourceStatus.ACTIVE))
    actives = list(reg.filter(status=SourceStatus.ACTIVE))
    assert {r.id for r in actives} == {"a", "c"}


def test_registry_active_to_retracted_is_allowed(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.ACTIVE))
    reg.update_status("a", SourceStatus.RETRACTED)
    assert reg.get("a").status is SourceStatus.RETRACTED


def test_registry_retracted_to_active_is_blocked(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.RETRACTED))
    with pytest.raises(InvalidTransitionError):
        reg.update_status("a", SourceStatus.ACTIVE)


def test_registry_pending_review_can_only_move_to_active_or_retracted(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.PENDING_REVIEW))
    with pytest.raises(InvalidTransitionError):
        reg.update_status("a", SourceStatus.SUPERSEDED)
    reg.update_status("a", SourceStatus.ACTIVE)
    assert reg.get("a").status is SourceStatus.ACTIVE


def test_registry_supersede_chains_records(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("arxiv:2408.13687v1", status=SourceStatus.ACTIVE))
    reg.append(_make("arxiv:2408.13687v2", status=SourceStatus.ACTIVE))
    reg.supersede(old_id="arxiv:2408.13687v1", new_id="arxiv:2408.13687v2")
    old = reg.get("arxiv:2408.13687v1")
    new = reg.get("arxiv:2408.13687v2")
    assert old.status is SourceStatus.SUPERSEDED
    assert old.superseded_by == "arxiv:2408.13687v2"
    assert new.supersedes == "arxiv:2408.13687v1"


def test_registry_compact_candidates_returns_old_retracted_and_superseded(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("old-retracted", status=SourceStatus.RETRACTED))
    reg.append(_make("old-superseded", status=SourceStatus.SUPERSEDED))
    reg.append(_make("active-keep", status=SourceStatus.ACTIVE))
    # Treat all records' last_checked_at as 2026-01-01; "now" = 2026-05-29 → 148 days.
    candidates = list(reg.compact_candidates(older_than_days=90, now=_utc(2026, 5, 29)))
    assert {r.id for r in candidates} == {"old-retracted", "old-superseded"}


def test_registry_compact_excludes_recent_records(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("recent-retracted", status=SourceStatus.RETRACTED))  # ingested 2026-01-01
    candidates = list(reg.compact_candidates(older_than_days=365, now=_utc(2026, 5, 29)))
    assert list(candidates) == []
