"""Watchlist refresh: pull candidate sources from arxiv, github, and rss."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from chatqec.ingestion.watchlist import (
    ArxivWatch,
    GitHubWatchEntry,
    Watchlist,
)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    title: str
    authors: list[str]
    year: int | None
    source_type: str


def _arxiv_client() -> Any:
    import arxiv

    return arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=2)


def refresh_arxiv(wl: ArxivWatch, *, now: datetime, lookback_days: int = 7) -> Iterator[Candidate]:
    if not (wl.categories or wl.keywords or wl.authors):
        return
    import arxiv

    cutoff = now - timedelta(days=lookback_days)
    queries: list[str] = []
    queries.extend(f"cat:{cat}" for cat in wl.categories)
    queries.extend(f'ti:"{kw}"' for kw in wl.keywords)
    queries.extend(f'au:"{auth}"' for auth in wl.authors)
    query = " OR ".join(queries) if queries else "cat:quant-ph"
    client = _arxiv_client()
    search = arxiv.Search(
        query=query,
        max_results=30,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    for r in client.results(search):
        if r.published.replace(tzinfo=None) < cutoff.replace(tzinfo=None):
            break
        m = re.search(r"abs/([^\s]+)", r.entry_id)
        arxiv_id = m.group(1) if m else r.entry_id
        yield Candidate(
            source_id=f"arxiv:{arxiv_id}",
            title=r.title.replace("\n", " ").strip(),
            authors=[a.name for a in r.authors],
            year=r.published.year,
            source_type="arxiv",
        )


def refresh_github(
    repos: Iterable[GitHubWatchEntry], *, now: datetime, lookback_days: int = 7
) -> Iterator[Candidate]:
    cutoff = now - timedelta(days=lookback_days)
    for entry in repos:
        url = f"https://api.github.com/repos/{entry.repo}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue
        pushed_at = data.get("pushed_at")
        if not pushed_at:
            continue
        try:
            pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if pushed < cutoff:
            continue
        commit_token = pushed_at[:10].replace("-", "")
        yield Candidate(
            source_id=f"github:{entry.repo}@{commit_token}",
            title=entry.repo,
            authors=[],
            year=None,
            source_type="github",
        )


def refresh_rss(
    feeds: Iterable[Any], *, now: datetime, lookback_days: int = 7
) -> Iterator[Candidate]:
    """RSS fetching is deferred — feedparser integration lands as a follow-up patch."""
    return
    yield  # pragma: no cover  # unreachable, keeps return type as Iterator


def refresh_all(wl: Watchlist, *, now: datetime) -> Iterator[Candidate]:
    yield from refresh_arxiv(wl.arxiv, now=now)
    yield from refresh_github(wl.github, now=now)
    yield from refresh_rss(wl.rss, now=now)
