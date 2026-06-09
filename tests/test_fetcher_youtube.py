"""YouTube transcript chunker - 60-second window aggregation."""

import json
from pathlib import Path

from chatqec.ingestion.fetchers.youtube import (
    YouTubeMetadata,
    transcript_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "youtube_transcript.json"


def test_transcript_to_chunks_groups_by_60s_window():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(
        video_id="abc123",
        title="Surface code 101",
        channel="QECChannel",
    )
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"], window_seconds=60)
    # 0s, 5s, 12s → window 0; 65s → window 1; 130s → window 2
    assert len(chunks) == 3
    # First chunk contains everything from 0-60s
    assert "Welcome" in chunks[0].text
    assert "MWPM" not in chunks[0].text
    # Second window has the MWPM line
    assert "MWPM" in chunks[1].text


def test_transcript_chunk_anchor_is_timestamp():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(video_id="abc123", title="t", channel="c")
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"], window_seconds=60)
    assert chunks[0].anchor == "?t=0s"
    assert chunks[1].anchor == "?t=60s"


def test_transcript_chunk_source_id_uses_video_id():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(video_id="abc123XYZ", title="t", channel="c")
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"], window_seconds=60)
    assert chunks[0].source_id == "youtube:abc123XYZ"


def test_empty_transcript_yields_no_chunks():
    md = YouTubeMetadata(video_id="x", title="t", channel="c")
    assert transcript_to_chunks(md, transcript_events=[], window_seconds=60) == []
