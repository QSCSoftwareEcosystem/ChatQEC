"""Canonical loader tests — scans knowledge/canonical/ and validates cross-refs."""

from pathlib import Path

import pytest

from chatqec.canonical.loader import (
    CrossReferenceError,
    DuplicateAliasError,
    DuplicateSlugError,
    load_canonical_directory,
)


def _write_page(
    path: Path, slug: str, aliases: list[str] | None = None, see_also: list[str] | None = None
) -> None:
    aliases = aliases or []
    see_also = see_also or []
    path.write_text(
        f"""---
topic_slug: {slug}
title: "{slug}"
aliases: {aliases!r}
see_also: {see_also!r}
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["m"]
---
# {slug}

body for {slug}
""",
        encoding="utf-8",
    )


def test_load_canonical_directory_returns_all_pages(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a")
    _write_page(tmp_path / "b.md", "b")
    pages = load_canonical_directory(tmp_path)
    assert {p.frontmatter.topic_slug for p in pages} == {"a", "b"}


def test_load_canonical_directory_skips_underscored_files(tmp_path: Path):
    # Files starting with _ (e.g. _roster.yaml, _index.md) are ignored
    _write_page(tmp_path / "a.md", "a")
    (tmp_path / "_index.md").write_text("# index", encoding="utf-8")
    (tmp_path / "_roster.yaml").write_text("foo: bar\n", encoding="utf-8")
    pages = load_canonical_directory(tmp_path)
    assert len(pages) == 1


def test_load_rejects_duplicate_slug(tmp_path: Path):
    _write_page(tmp_path / "a.md", "x")
    _write_page(tmp_path / "b.md", "x")  # same slug, different file
    with pytest.raises(DuplicateSlugError):
        load_canonical_directory(tmp_path)


def test_load_rejects_alias_colliding_with_existing_slug(tmp_path: Path):
    _write_page(tmp_path / "a.md", "surface-code", aliases=["color-code"])
    _write_page(tmp_path / "b.md", "color-code")
    with pytest.raises(DuplicateAliasError):
        load_canonical_directory(tmp_path)


def test_load_rejects_unresolved_see_also(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a", see_also=["nonexistent-slug"])
    _write_page(tmp_path / "b.md", "b")
    with pytest.raises(CrossReferenceError, match="nonexistent-slug"):
        load_canonical_directory(tmp_path)


def test_load_allows_see_also_pointing_at_alias(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a", see_also=["toric-code"])
    _write_page(tmp_path / "b.md", "surface-code", aliases=["toric-code"])
    # Should NOT raise — see_also resolution checks slugs + aliases.
    pages = load_canonical_directory(tmp_path)
    assert len(pages) == 2
