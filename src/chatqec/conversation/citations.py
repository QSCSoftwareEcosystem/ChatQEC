"""Parse footnote markers [^N] out of Sonnet's text and resolve to Citation objects."""

from __future__ import annotations

import re
from typing import Any

from chatqec.api_types import Citation
from chatqec.retrieval.hybrid import RetrievedChunk

_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")

# A line that begins a model-written "Sources:" list (optionally markdown-wrapped).
_SOURCES_LINE_RE = re.compile(r"^\s*(?:[#*]+\s*)*sources\s*:", re.IGNORECASE)


def strip_trailing_sources(text: str) -> str:
    """Remove a trailing model-written "Sources:" list, keeping inline [^n] markers.

    The CLI/app renders its own linked source list from the extracted citations,
    so the model's plain-text list is redundant. Inline footnote markers in the
    body are left untouched.
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if _SOURCES_LINE_RE.match(lines[i]):
            return "\n".join(lines[:i]).rstrip()
    return text


def _resolve_url(payload: dict[str, Any], source_id: str) -> str:
    """Pick the best URL we can from a chunk's payload, deriving arXiv abs URLs."""
    for key in ("url", "meta_url", "pdf_url", "meta_pdf_url"):
        v = payload.get(key)
        if v:
            return str(v)
    if source_id.startswith("arxiv:"):
        return f"https://arxiv.org/abs/{source_id.removeprefix('arxiv:')}"
    return ""


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
                url=_resolve_url(c.payload, c.source_id),
                marker=n,
            )
        )
    return out
