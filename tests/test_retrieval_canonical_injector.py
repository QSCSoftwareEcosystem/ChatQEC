"""Canonical force-injection by topic match (string-based in M6; LLM-based in M7)."""

from datetime import date

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage
from chatqec.retrieval.canonical_injector import (
    CanonicalIndex,
    match_topics,
)


def _page(slug: str, aliases: list[str] | None = None) -> CanonicalPage:
    return CanonicalPage(
        frontmatter=CanonicalFrontmatter(
            topic_slug=slug,
            title=slug.replace("-", " ").title(),
            aliases=aliases or [],
            see_also=[],
            authority_tier="canonical",
            last_reviewed=date(2026, 5, 29),
            maintainers=["m"],
        ),
        body=f"Body of {slug}",
    )


def test_index_matches_slug_in_query_substring():
    idx = CanonicalIndex([_page("surface-code"), _page("color-code")])
    hits = match_topics("Compare the surface code threshold", idx)
    assert "surface-code" in hits


def test_index_matches_alias_in_query():
    idx = CanonicalIndex([_page("surface-code", aliases=["toric code"])])
    hits = match_topics("How does the toric code work?", idx)
    assert "surface-code" in hits


def test_index_is_case_insensitive():
    idx = CanonicalIndex([_page("surface-code", aliases=["Toric Code"])])
    hits = match_topics("explain TORIC code", idx)
    assert "surface-code" in hits


def test_index_returns_empty_for_unrelated_query():
    idx = CanonicalIndex([_page("surface-code")])
    hits = match_topics("What's for lunch?", idx)
    assert hits == []


def test_index_handles_punctuation_in_query():
    idx = CanonicalIndex([_page("surface-code")])
    hits = match_topics("Surface code, please?", idx)
    assert "surface-code" in hits


def test_index_dedups_repeated_matches():
    idx = CanonicalIndex([_page("surface-code", aliases=["toric code", "planar code"])])
    hits = match_topics("toric code AND planar code AND surface code", idx)
    assert hits == ["surface-code"]
