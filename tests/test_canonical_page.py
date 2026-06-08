"""CanonicalPage schema + frontmatter parsing tests."""

from datetime import date

import pytest
from pydantic import ValidationError

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage, parse_page


def test_frontmatter_accepts_valid_kebab_slug():
    fm = CanonicalFrontmatter(
        topic_slug="surface-code",
        title="Surface Code",
        aliases=["toric code", "planar code"],
        see_also=["stabilizer-formalism", "color-code"],
        authority_tier="canonical",
        last_reviewed=date(2026, 5, 29),
        maintainers=["sharmin"],
    )
    assert fm.topic_slug == "surface-code"


def test_frontmatter_rejects_uppercase_slug():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="Surface-Code",
            title="x",
            authority_tier="canonical",
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_frontmatter_rejects_underscore_in_slug():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="surface_code",
            title="x",
            authority_tier="canonical",
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_frontmatter_requires_canonical_authority_tier():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="x",
            title="x",
            authority_tier="preprint",  # not allowed for canonical pages
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_parse_page_splits_frontmatter_and_body():
    text = """---
topic_slug: surface-code
title: "Surface Code"
aliases: ["toric code"]
see_also: ["stabilizer-formalism"]
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["sharmin"]
---
# Surface Code

The surface code is a topological CSS code.

## Stabilizers

X-type plaquettes and Z-type stars.
"""
    page = parse_page(text)
    assert isinstance(page, CanonicalPage)
    assert page.frontmatter.topic_slug == "surface-code"
    assert "topological" in page.body.lower()
    # The title heading should appear in the body
    assert page.body.lstrip().startswith("# Surface Code")


def test_parse_page_rejects_missing_frontmatter():
    text = "# Surface Code\n\nNo frontmatter here.\n"
    with pytest.raises(ValueError, match="frontmatter"):
        parse_page(text)


def test_parse_page_rejects_unterminated_frontmatter():
    text = "---\ntopic_slug: x\n# missing closing fence\n"
    with pytest.raises(ValueError):
        parse_page(text)
