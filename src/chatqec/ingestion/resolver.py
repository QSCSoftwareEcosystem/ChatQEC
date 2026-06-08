"""arXiv title-and-author resolver. See spec §7.4 and A.10.

LLM-generated arXiv IDs are unreliable (~14% accurate per the verification
pass during brainstorming). The correct ingestion pattern is to search by
title + author and pick the unique best match.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import arxiv


class ResolverError(Exception):
    pass


class NoMatchError(ResolverError):
    pass


class AmbiguousResolutionError(ResolverError):
    pass


@dataclass(frozen=True)
class ResolverHit:
    arxiv_id: str
    title: str
    first_author: str
    year: int


# Scoring constants (spec §7.4 / A.10)
_SCORE_TITLE_SUBSTRING = 5.0  # full keyword phrase found in title
_SCORE_TITLE_WORD = 1.0  # per matching content word (>3 chars)
_SCORE_AUTHOR = 2.0  # query last name matches result last name
_SCORE_YEAR_EXACT = 2.0  # publication year matches expected
_SCORE_YEAR_OFF1 = 0.5  # publication year off by one (drift)
_MIN_WORD_LEN = 3  # words ≤ this length are ignored in overlap scoring
_NO_MATCH_THRESHOLD = 4.0  # top score must exceed this
_AMBIGUITY_GAP = 1.0  # top two scores must differ by more than this
_MIN_CANDIDATES = 2  # need at least this many to check ambiguity

# Reusable client; arxiv lib internally rate-limits to 3 req/s.
_CLIENT = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)


class _SearchWrapper:
    """Thin wrapper around arxiv.Search that exposes a `.results()` method
    backed by the module-level *_CLIENT*.  This indirection keeps the public
    surface testable: tests monkeypatch *_arxiv_search* and return a mock
    whose `.results()` yields fake entries, while production code goes through
    the rate-limited *_CLIENT*.
    """

    def __init__(self, search: Any) -> None:
        self._search = search

    def results(self) -> Iterator[Any]:
        return _CLIENT.results(self._search)  # type: ignore[no-any-return]


def _arxiv_search(query: str, max_results: int = 10) -> _SearchWrapper:
    return _SearchWrapper(arxiv.Search(query=query, max_results=max_results))


def _id_from_entry(entry_id: str, strip_version: bool) -> str:
    m = re.search(r"abs/([^v\s]+)(v\d+)?", entry_id)
    if not m:
        raise ResolverError(f"could not parse arxiv id from {entry_id!r}")
    base = m.group(1)
    version = m.group(2) or ""
    return base if strip_version else (base + version)


def _score(
    result_title: str,
    result_authors_lower: list[str],
    result_year: int,
    title_keywords: str,
    authors: list[str],
    expected_year: int | None,
) -> float:
    """Higher = better match."""
    score = 0.0
    title_l = result_title.lower()
    kw_l = title_keywords.lower()
    # title keyword full substring match is the strongest signal
    if kw_l in title_l:
        score += _SCORE_TITLE_SUBSTRING
    # per-word overlap
    kw_words = [w for w in kw_l.split() if len(w) > _MIN_WORD_LEN]
    title_words = set(title_l.split())
    score += sum(_SCORE_TITLE_WORD for w in kw_words if w in title_words)
    # author match — query last name matches result author last name
    for a in authors:
        last = a.split()[-1].lower()
        if any(last == ra.split()[-1] for ra in result_authors_lower):
            score += _SCORE_AUTHOR
    # year match
    if expected_year is not None:
        if result_year == expected_year:
            score += _SCORE_YEAR_EXACT
        elif abs(result_year - expected_year) == 1:
            score += _SCORE_YEAR_OFF1  # publish year drift
    return score


def resolve_arxiv_paper(
    *,
    title_keywords: str,
    authors: list[str],
    expected_year: int | None = None,
    strip_version: bool = False,
) -> ResolverHit:
    """Return the unique best-matching arXiv paper for the given title+authors.

    Raises:
        NoMatchError: no result above the minimum quality threshold.
        AmbiguousResolutionError: top two candidates are within 1.0 score points.
    """
    query = f'ti:"{title_keywords}" AND au:"{authors[0]}"' if authors else f'ti:"{title_keywords}"'
    search = _arxiv_search(query, max_results=10)
    results = list(search.results())

    scored: list[tuple[float, Any]] = []
    for r in results:
        author_names = [a.name.lower() for a in r.authors]
        s = _score(r.title, author_names, r.published.year, title_keywords, authors, expected_year)
        scored.append((s, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < _NO_MATCH_THRESHOLD:
        raise NoMatchError(
            f"no arXiv result above quality threshold for title={title_keywords!r}, authors={authors!r}"
        )
    if len(scored) >= _MIN_CANDIDATES and scored[0][0] - scored[1][0] < _AMBIGUITY_GAP:
        raise AmbiguousResolutionError(
            f"two candidates tied: {scored[0][1].title!r} vs {scored[1][1].title!r}"
        )

    top = scored[0][1]
    return ResolverHit(
        arxiv_id=_id_from_entry(top.entry_id, strip_version=strip_version),
        title=top.title,
        first_author=top.authors[0].name if top.authors else "",
        year=top.published.year,
    )
