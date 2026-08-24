"""QAppsWiki sections as retrieval candidates, fetched over MCP.

The wiki serves its own BM25 search (`search_pages`), so ChatQEC keeps no copy of
the corpus and always sees the current working tree. Hits come back as ordinary
RetrievedChunks and are reranked against Qdrant results, so the cross-encoder
decides what actually reaches the synthesizer.

Best-effort by construction: a missing server, a timeout, or a malformed payload
returns [] and the answer proceeds from the corpus alone.
"""

from __future__ import annotations

import logging
from typing import Any

from chatqec.retrieval.hybrid import RetrievedChunk
from chatqec.trace import trace

logger = logging.getLogger(__name__)

_TOOL = "search_pages"
_CURATED_STATUSES = frozenset({"source-backed", "verified"})
# Zoo-imported pages state: "Text is reused under CC-BY-SA; attribute the Error
# Correction Zoo". citations.py renders only title and url, so the attribution has
# to travel in the title or a reader never sees it.
_ZOO_HOST = "errorcorrectionzoo.org"
_ZOO_ATTRIBUTION = "CC-BY-SA Error Correction Zoo"


def _attribution(sources: list[str]) -> str:
    return _ZOO_ATTRIBUTION if any(_ZOO_HOST in str(s) for s in sources) else ""


def _tier(provenance_status: str) -> str:
    return "wiki_curated" if provenance_status in _CURATED_STATUSES else "wiki_unverified"


def _to_chunk(row: dict[str, Any]) -> RetrievedChunk | None:
    node_id = str(row.get("node_id") or "")
    text = str(row.get("text") or "")
    if not node_id or not text:
        return None
    heading = str(row.get("heading") or "")
    source_id = f"qappswiki:{node_id}"
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    sources = list(row.get("sources") or [])
    attribution = _attribution(sources)
    base_title = str(row.get("title") or node_id)
    suffix = f"QAppsWiki, {attribution}" if attribution else "QAppsWiki"
    return RetrievedChunk(
        chunk_id=f"{source_id}#{heading}",
        source_id=source_id,
        text=text,
        title=f"{base_title} ({suffix})",
        authority_tier=_tier(str(row.get("provenance_status") or "")),
        score=score,
        payload={
            "url": str(row.get("url") or ""),
            "path": str(row.get("path") or ""),
            "heading": heading,
            "sources": sources,
            "attribution": attribution,
            "provenance_status": str(row.get("provenance_status") or ""),
        },
    )


def wiki_chunks(
    client: Any, query: str, *, k: int = 8, domain: str | None = None
) -> list[RetrievedChunk]:
    """Search QAppsWiki over MCP and return chunks. Never raises."""
    if client is None or not query:
        return []
    args: dict[str, Any] = {"query": query, "k": k, "domain": domain}
    try:
        payload = client.call_tool(_TOOL, args)
    except Exception:  # missing tool, dead server, timeout — all non-fatal
        logger.warning("QAppsWiki search failed; answering from the corpus alone", exc_info=True)
        return []

    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    chunks = [c for c in (_to_chunk(r) for r in rows if isinstance(r, dict)) if c is not None]
    trace("wiki.search", query=query, k=k, returned=len(chunks))
    return chunks
