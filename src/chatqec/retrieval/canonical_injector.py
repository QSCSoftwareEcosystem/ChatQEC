"""Canonical force-injection: identify canonical pages relevant to a query.

M6 uses simple substring matching against page slugs and aliases. M7 may
upgrade this to an LLM-based topic classifier (Haiku), but the interface
stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chatqec.canonical.page import CanonicalPage

_NON_WORD = re.compile(r"[^a-z0-9\s-]+")


@dataclass(frozen=True)
class CanonicalIndex:
    """In-memory index of canonical pages by slug + alias.

    `pages` is the full list of loaded canonical pages; we build a lookup of
    normalized terms to slugs at construction time.
    """

    pages: list[CanonicalPage]

    def lookup(self) -> dict[str, str]:
        """Map normalized term → canonical topic_slug."""
        m: dict[str, str] = {}
        for p in self.pages:
            slug = p.frontmatter.topic_slug
            m[_normalize(slug.replace("-", " "))] = slug
            m[_normalize(p.frontmatter.title)] = slug
            for alias in p.frontmatter.aliases:
                m[_normalize(alias)] = slug
        return m


def _normalize(s: str) -> str:
    return _NON_WORD.sub("", s.lower()).strip()


def match_topics(query: str, index: CanonicalIndex) -> list[str]:
    """Return canonical slugs whose term appears in the query, in first-match order."""
    qn = _normalize(query)
    hits: list[str] = []
    seen: set[str] = set()
    # Match longest terms first so "surface code" wins over "code"
    sorted_terms = sorted(index.lookup().items(), key=lambda kv: len(kv[0]), reverse=True)
    for term, slug in sorted_terms:
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", qn) and slug not in seen:
            hits.append(slug)
            seen.add(slug)
    return hits
