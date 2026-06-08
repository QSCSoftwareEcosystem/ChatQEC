"""arXiv resolver — the §7.4 mandatory pattern.

Never trust an LLM-claimed arXiv ID. Always resolve by title + author via the
arXiv search API. The verification finding in spec A.10 showed 71% of LLM IDs
point to wrong papers; this resolver is the architectural fix.
"""

from unittest.mock import MagicMock

import pytest

from chatqec.ingestion.resolver import (
    AmbiguousResolutionError,
    NoMatchError,
    ResolverHit,
    resolve_arxiv_paper,
)


class _FakeResult:
    def __init__(self, entry_id: str, title: str, authors: list[str], year: int) -> None:
        self.entry_id = entry_id  # e.g. "http://arxiv.org/abs/2408.13687v1"
        self.title = title
        self.authors = [MagicMock(name=a) for a in authors]
        for a, name in zip(self.authors, authors, strict=True):
            a.name = name
        self.published = MagicMock(year=year)


def _patch_search(monkeypatch, results: list[_FakeResult]) -> None:
    mock_search = MagicMock()
    mock_search.return_value.results.return_value = iter(results)
    monkeypatch.setattr("chatqec.ingestion.resolver._arxiv_search", mock_search)


def test_resolver_returns_unique_match(monkeypatch):
    _patch_search(
        monkeypatch,
        [
            _FakeResult(
                "http://arxiv.org/abs/2408.13687v1",
                "Quantum error correction below the surface code threshold",
                ["Rajeev Acharya", "Et al"],
                2024,
            )
        ],
    )
    hit = resolve_arxiv_paper(
        title_keywords="below the surface code threshold",
        authors=["Acharya"],
    )
    assert isinstance(hit, ResolverHit)
    assert hit.arxiv_id == "2408.13687v1"
    assert "Acharya" in hit.first_author


def test_resolver_strips_version_suffix_when_asked(monkeypatch):
    _patch_search(
        monkeypatch,
        [
            _FakeResult(
                "http://arxiv.org/abs/2408.13687v3",
                "Quantum error correction below the surface code threshold",
                ["Rajeev Acharya"],
                2024,
            )
        ],
    )
    hit = resolve_arxiv_paper(
        title_keywords="surface code threshold",
        authors=["Acharya"],
        strip_version=True,
    )
    assert hit.arxiv_id == "2408.13687"


def test_resolver_raises_no_match(monkeypatch):
    _patch_search(monkeypatch, [])
    with pytest.raises(NoMatchError):
        resolve_arxiv_paper(title_keywords="nonsense paper", authors=["Nobody"])


def test_resolver_raises_ambiguous_when_multiple_strong_matches(monkeypatch):
    _patch_search(
        monkeypatch,
        [
            _FakeResult(
                "http://arxiv.org/abs/2207.06431v1",
                "Suppressing quantum errors by scaling a surface code logical qubit",
                ["Google Quantum AI"],
                2022,
            ),
            _FakeResult(
                "http://arxiv.org/abs/2408.13687v1",
                "Quantum error correction below the surface code threshold",
                ["Rajeev Acharya"],
                2024,
            ),
        ],
    )
    with pytest.raises(AmbiguousResolutionError):
        # Both contain "surface code" and "Google" partially matches both; no
        # clear winner without disambiguating by year.
        resolve_arxiv_paper(title_keywords="surface code", authors=["Google"])


def test_resolver_disambiguates_by_year(monkeypatch):
    _patch_search(
        monkeypatch,
        [
            _FakeResult(
                "http://arxiv.org/abs/2207.06431v1",
                "Suppressing quantum errors by scaling a surface code logical qubit",
                ["Google Quantum AI"],
                2022,
            ),
            _FakeResult(
                "http://arxiv.org/abs/2408.13687v1",
                "Quantum error correction below the surface code threshold",
                ["Rajeev Acharya"],
                2024,
            ),
        ],
    )
    hit = resolve_arxiv_paper(
        title_keywords="surface code threshold",
        authors=["Acharya"],
        expected_year=2024,
    )
    assert hit.arxiv_id == "2408.13687v1"
