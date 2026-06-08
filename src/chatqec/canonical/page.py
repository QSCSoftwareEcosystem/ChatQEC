"""Canonical wiki page schema and frontmatter parser.

Canonical pages live in knowledge/canonical/<slug>.md and are the only sources
authored by humans rather than fetched. They get authority_tier="canonical"
and are force-injected into context when the query matches their topic_slug
or aliases (see spec §5.2 and §8.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class CanonicalFrontmatter(BaseModel):
    """YAML frontmatter of a canonical wiki page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_slug: str
    title: str
    aliases: list[str] = Field(default_factory=list)
    see_also: list[str] = Field(default_factory=list)
    authority_tier: str
    last_reviewed: date
    maintainers: list[str]

    @field_validator("topic_slug")
    @classmethod
    def _kebab_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(f"topic_slug must be kebab-case (a-z, 0-9, hyphen): {v!r}")
        return v

    @field_validator("see_also", "aliases")
    @classmethod
    def _no_blanks(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("entries must be non-empty strings")
        return v

    @field_validator("authority_tier")
    @classmethod
    def _must_be_canonical(cls, v: str) -> str:
        if v != "canonical":
            raise ValueError(f"canonical pages must have authority_tier='canonical', got {v!r}")
        return v


@dataclass(frozen=True)
class CanonicalPage:
    """A parsed canonical wiki page = frontmatter + raw markdown body."""

    frontmatter: CanonicalFrontmatter
    body: str


def parse_page(text: str) -> CanonicalPage:
    """Parse a canonical page from raw markdown text."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "canonical page must start with YAML frontmatter delimited by '---' fences"
        )
    fm_text, body = match.group(1), match.group(2)
    fm_data = yaml.safe_load(fm_text)
    if not isinstance(fm_data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return CanonicalPage(
        frontmatter=CanonicalFrontmatter(**fm_data),
        body=body,
    )
