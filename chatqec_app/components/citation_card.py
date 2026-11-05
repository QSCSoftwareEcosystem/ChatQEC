"""Renders citations as expandable cards, one card per source.

An answer often cites several chunks of the same paper ([^1], [^4], [^9] can all be
arXiv:2408.13687). One card per chunk repeated the same title down the page, so chunks are
grouped by source and the card lists every footnote marker that points to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from chatqec.api import Citation


@dataclass
class SourceGroup:
    source_id: str
    title: str
    url: str
    markers: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        marks = "".join(f"[^{m}]" for m in self.markers if m)
        return f"{marks} [{self.source_id}] {self.title}".strip()


def group_by_source(citations: list[Citation]) -> list[SourceGroup]:
    """Group citations by source, in the order each source is first cited."""
    groups: dict[str, SourceGroup] = {}
    for c in citations:
        g = groups.setdefault(c.source_id, SourceGroup(c.source_id, c.title, c.url))
        g.markers.append(c.marker)
        g.chunk_ids.append(c.chunk_id)
    return list(groups.values())


def render_all(citations: list[Citation]) -> None:
    for g in group_by_source(citations):
        with st.expander(g.label, expanded=False):
            if g.url:
                st.write(f"[link]({g.url})")
            st.caption("chunks: " + ", ".join(f"`{cid}`" for cid in g.chunk_ids))


def render(citation: Citation) -> None:
    """One citation on its own card (kept for callers outside the Chat page)."""
    render_all([citation])
