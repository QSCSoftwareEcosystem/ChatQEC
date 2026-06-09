"""Parse footnote markers [^N] out of Sonnet's text and resolve to Citation objects."""

from __future__ import annotations

import re

from chatqec.api_types import Citation
from chatqec.retrieval.hybrid import RetrievedChunk

_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")


def extract_citations(
    answer_text: str,
    context_chunks: list[RetrievedChunk],
) -> list[Citation]:
    seen: set[int] = set()
    ordered: list[int] = []
    for m in _FOOTNOTE_RE.finditer(answer_text):
        n = int(m.group(1))
        if 1 <= n <= len(context_chunks) and n not in seen:
            seen.add(n)
            ordered.append(n)
    out: list[Citation] = []
    for n in ordered:
        c = context_chunks[n - 1]
        out.append(
            Citation(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                title=c.title,
                url=str(c.payload.get("url", "")),
            )
        )
    return out
