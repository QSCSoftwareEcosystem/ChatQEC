"""YouTube transcript chunker.

The actual download uses `yt-dlp --write-auto-subs --skip-download` from a
higher-level orchestrator. This module operates on already-parsed events.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from chatqec.ingestion.chunk import Chunk


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    title: str
    channel: str


def transcript_to_chunks(
    md: YouTubeMetadata,
    *,
    transcript_events: Iterable[dict[str, object]],
    window_seconds: int = 60,
) -> list[Chunk]:
    events = list(transcript_events)
    if not events:
        return []

    source_id = f"youtube:{md.video_id}"
    buckets: dict[int, list[str]] = {}
    for ev in events:
        t_ms_obj = ev.get("tStartMs", 0)
        t_ms = int(cast(int, t_ms_obj))
        bucket = (t_ms // 1000) // window_seconds
        segs_obj = ev.get("segs", [])
        segs = cast(list[dict[str, str]], segs_obj)
        text_pieces = [s.get("utf8", "") for s in segs if isinstance(s, dict)]
        text = "".join(text_pieces).strip()
        if not text:
            continue
        buckets.setdefault(bucket, []).append(text)

    chunks: list[Chunk] = []
    for i, bucket in enumerate(sorted(buckets)):
        start_seconds = bucket * window_seconds
        text = " ".join(buckets[bucket])
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}#{i}",
                source_id=source_id,
                chunk_index=i,
                text=text,
                title=md.title,
                source_type="youtube",
                authority_tier="video",
                anchor=f"?t={start_seconds}s",
                metadata={"channel": md.channel, "start_seconds": start_seconds},
            )
        )
    return chunks
