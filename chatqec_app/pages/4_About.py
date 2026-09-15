"""About page — what's in the corpus, license, citation."""

import json
from pathlib import Path

import streamlit as st

from chatqec import __version__
from chatqec_app.utils import render_branding

render_branding()
st.title("About ChatQEC")

st.markdown(
    """
ChatQEC is an open-source research assistant for **Quantum Error Correction**.

It retrieves from a curated corpus of papers, textbooks, GitHub repos, blog posts,
and video transcripts, then answers your question via Claude Sonnet with inline
citations. It can also run small Stim circuits in a sandbox.

**Stack:** Python, Hydra, Qdrant, Anthropic Claude, Google Gemini, Stim, MCP.

**License:** Apache 2.0.

**Source:** [`gitlab.com/<USERNAME>/chatqec`](https://gitlab.com/<USERNAME>/chatqec)
and the companion [`chatqec-mcp-tools`](https://gitlab.com/<USERNAME>/chatqec-mcp-tools).
    """
)

reg_path = Path("sources/registry.jsonl")
if reg_path.exists():
    records = [json.loads(line) for line in reg_path.read_text().splitlines() if line.strip()]
    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    st.subheader("Corpus summary")
    st.write(f"📚 {len(records)} sources total")
    for t, n in sorted(by_type.items()):
        st.write(f"- {t}: {n}")

st.caption(f"chatqec v{__version__}")
