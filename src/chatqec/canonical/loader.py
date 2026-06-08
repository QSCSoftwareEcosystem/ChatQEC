"""Loader and cross-reference validator for the canonical wiki directory."""

from __future__ import annotations

from pathlib import Path

from chatqec.canonical.page import CanonicalPage, parse_page


class CanonicalLoadError(Exception):
    pass


class DuplicateSlugError(CanonicalLoadError):
    pass


class DuplicateAliasError(CanonicalLoadError):
    pass


class CrossReferenceError(CanonicalLoadError):
    pass


def load_canonical_directory(directory: Path) -> list[CanonicalPage]:
    """Scan a directory of canonical pages, parse, and validate cross-refs.

    Rules:
    - Files starting with '_' are ignored (e.g. _roster.yaml, _index.md).
    - All topic_slugs must be unique.
    - No alias may collide with another page's slug or another alias.
    - Every see_also entry must resolve to an existing slug or alias.
    """
    pages: list[CanonicalPage] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.startswith("_"):
            continue
        pages.append(parse_page(path.read_text(encoding="utf-8")))

    slugs: set[str] = set()
    alias_to_slug: dict[str, str] = {}
    for p in pages:
        slug = p.frontmatter.topic_slug
        if slug in slugs:
            raise DuplicateSlugError(slug)
        slugs.add(slug)
    for p in pages:
        for alias in p.frontmatter.aliases:
            if alias in slugs and alias != p.frontmatter.topic_slug:
                raise DuplicateAliasError(
                    f"alias {alias!r} on page {p.frontmatter.topic_slug!r} collides with slug {alias!r}"
                )
            if alias in alias_to_slug:
                raise DuplicateAliasError(
                    f"alias {alias!r} appears on both "
                    f"{alias_to_slug[alias]!r} and {p.frontmatter.topic_slug!r}"
                )
            alias_to_slug[alias] = p.frontmatter.topic_slug

    resolvable = slugs | set(alias_to_slug.keys())
    for p in pages:
        for ref in p.frontmatter.see_also:
            if ref not in resolvable:
                raise CrossReferenceError(
                    f"see_also entry {ref!r} on page {p.frontmatter.topic_slug!r} "
                    f"does not resolve to any slug or alias"
                )
    return pages
