"""Source registry model + I/O tests."""

import json
from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from chatqec.ingestion.registry import SourceRecord, SourceStatus, SourceType


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
