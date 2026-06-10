"""Watchlist refresh tests — fetcher logic mocked."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from chatqec.ingestion.watchlist import ArxivWatch, GitHubWatchEntry, Watchlist
from chatqec.ingestion.watchlist_refresh import (
    Candidate,
    refresh_all,
    refresh_arxiv,
    refresh_github,
)


def test_refresh_arxiv_yields_candidates_from_search(monkeypatch):
    fake_author = MagicMock()
    fake_author.name = "Acharya"
    fake_result = MagicMock()
    fake_result.entry_id = "http://arxiv.org/abs/2408.13687v1"
    fake_result.title = "Below threshold"
    fake_result.authors = [fake_author]
    fake_result.published = MagicMock(year=2024)
    fake_result.published.replace.return_value = datetime(2024, 8, 25)

    monkeypatch.setattr(
        "chatqec.ingestion.watchlist_refresh._arxiv_client",
        lambda: MagicMock(results=lambda search: iter([fake_result])),
    )
    cands = list(
        refresh_arxiv(
            ArxivWatch(
                categories=["quant-ph"],
                keywords=["surface code"],
                authors=[],
            ),
            now=datetime(2024, 9, 1, tzinfo=UTC),
        )
    )
    assert len(cands) == 1
    assert cands[0].source_id == "arxiv:2408.13687v1"


def test_refresh_github_yields_repo_candidates(monkeypatch):
    fake_meta = {
        "default_branch": "main",
        "pushed_at": "2026-05-25T00:00:00Z",
        "stargazers_count": 500,
        "license": {"spdx_id": "Apache-2.0"},
    }
    fake_get = MagicMock()
    fake_get.return_value.json.return_value = fake_meta
    fake_get.return_value.raise_for_status.return_value = None
    monkeypatch.setattr("chatqec.ingestion.watchlist_refresh.requests.get", fake_get)

    cands = list(
        refresh_github(
            [GitHubWatchEntry(repo="quantumlib/Stim")],
            now=datetime(2026, 5, 29, tzinfo=UTC),
        )
    )
    assert len(cands) == 1
    assert cands[0].source_id.startswith("github:quantumlib/Stim@")


def test_refresh_all_aggregates_sources(monkeypatch):
    monkeypatch.setattr(
        "chatqec.ingestion.watchlist_refresh.refresh_arxiv",
        lambda wl, *, now: iter(
            [Candidate(source_id="arxiv:a", title="t", authors=[], year=2024, source_type="arxiv")]
        ),
    )
    monkeypatch.setattr(
        "chatqec.ingestion.watchlist_refresh.refresh_github",
        lambda repos, *, now: iter(
            [
                Candidate(
                    source_id="github:b", title="t", authors=[], year=2024, source_type="github"
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "chatqec.ingestion.watchlist_refresh.refresh_rss",
        lambda feeds, *, now: iter([]),
    )

    wl = Watchlist()
    cands = list(refresh_all(wl, now=datetime(2026, 5, 29, tzinfo=UTC)))
    assert {c.source_id for c in cands} == {"arxiv:a", "github:b"}
