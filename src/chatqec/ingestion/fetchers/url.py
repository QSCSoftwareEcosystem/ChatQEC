"""URL/blog fetcher.

A higher-level orchestrator uses `trafilatura` (substitute for the plan's
`defuddle`, which is Node-only) to convert raw HTML → markdown, then passes
the markdown text to `markdown_to_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.ids import url_id


@dataclass(frozen=True)
class URLMetadata:
    url: str
    title: str
    publish_date: str  # "YYYY-MM-DD" or ""


def markdown_to_chunks(
    md: URLMetadata,
    *,
    markdown_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = url_id(md.url)
    pieces = chunk_text(markdown_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=p,
            title=md.title,
            source_type="blog",
            authority_tier="blog",
            anchor=f"#chunk-{i}",
            metadata={"url": md.url, "publish_date": md.publish_date},
        )
        for i, p in enumerate(pieces)
    ]
