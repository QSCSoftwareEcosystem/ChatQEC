"""Watchlist page — read-only view + 'Submit a source' link."""

from pathlib import Path

import streamlit as st

st.title("Watchlist")

wl_path = Path("sources/watchlist.yaml")
if not wl_path.exists():
    st.info("No watchlist yet.")
    st.stop()

st.code(wl_path.read_text(encoding="utf-8"), language="yaml")

st.markdown("---")
st.subheader("Submit a source")
st.write(
    "Open a GitLab issue with the source URL and a one-line justification. "
    "Maintainers review and merge — the weekly refresh job picks it up after that."
)
gitlab_repo = "<USERNAME>/chatqec"  # replace at runtime via env var if desired
st.link_button(
    "Submit a source",
    f"https://gitlab.com/{gitlab_repo}/-/issues/new?issuable_template=add_source",
)
