"""Tavily-backed web search — the corpus fallback source.

A thin, dependency-light client for Tavily's /search endpoint (called with the
already-present `requests`). The chosen backend returns clean extracted page
*content* per result (not just SERP links), so answers can be grounded and cited
without a separate fetch step. The function is best-effort: any failure — missing
key, network error, non-200, malformed body — returns an empty list so the caller
degrades to corpus-only behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

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
        out.append(
            WebResult(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                content=str(r.get("content", "")),
                score=float(r.get("score", 0.0) or 0.0),
            )
        )
    return out
