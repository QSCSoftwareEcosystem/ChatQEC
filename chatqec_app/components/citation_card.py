"""Renders a Citation as an expandable card."""

import streamlit as st

from chatqec.api import Citation


def render(citation: Citation) -> None:
    with st.expander(f"[{citation.source_id}] {citation.title}", expanded=False):
        if citation.url:
            st.write(f"[link]({citation.url})")
        st.caption(f"chunk: `{citation.chunk_id}`")
