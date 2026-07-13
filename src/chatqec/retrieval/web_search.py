"""Tavily-backed web search — the corpus fallback source.

A thin, dependency-light client for Tavily's /search endpoint (called with the
already-present `requests`). The chosen backend returns clean extracted page
*content* per result (not just SERP links), so answers can be grounded and cited
without a separate fetch step. The function is best-effort: any failure — missing
key, network error, non-200, malformed body — returns an empty list so the caller
degrades to corpus-only behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import requests

from chatqec.ingestion.hashing import sha256_text
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk

_TAVILY_URL = "https://api.tavily.com/search"
_HTTP_OK = 200


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float


def web_search(query: str, *, k: int, api_key: str, timeout: float = 15.0) -> list[WebResult]:
    """Search the web via Tavily; return up to ``k`` results. Never raises."""
    if not api_key:
        return []
    try:
        resp = requests.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": k,
                "search_depth": "basic",
            },
            timeout=timeout,
        )
        if resp.status_code != _HTTP_OK:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    out: list[WebResult] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            score = float(r.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(
            WebResult(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                content=str(r.get("content", "")),
                score=score,
            )
        )
    return out


def build_web_chunks(
    question: str,
    *,
    retrieval: RetrievalResult,
    web_searcher: Callable[[str, int], list[WebResult]] | None,
    threshold: float,
    web_k: int,
) -> list[RetrievedChunk]:
    """Return web-tier chunks when the corpus retrieval is weak, else ``[]``.

    Deterministic gate: search only when a searcher is configured, no canonical
    topic was force-injected, and the retrieval is weak (no chunks, or the best
    chunk score is below ``threshold``). Never raises.
    """
    if web_searcher is None:
        return []
    if retrieval.matched_topics:
        return []
    top = max((c.score for c in retrieval.chunks), default=0.0)
    if retrieval.chunks and top >= threshold:
        return []
    try:
        results = web_searcher(question, web_k)
    except Exception:  # searcher is best-effort; never break the answer path
        return []
    chunks: list[RetrievedChunk] = []
    for r in results:
        sid = f"web:{sha256_text(r.url)[:8]}"
        chunks.append(
            RetrievedChunk(
                chunk_id=f"{sid}#0",
                source_id=sid,
                text=r.content,
                title=r.title,
                authority_tier="web",
                score=r.score,
                payload={"url": r.url},
            )
        )
    return chunks
