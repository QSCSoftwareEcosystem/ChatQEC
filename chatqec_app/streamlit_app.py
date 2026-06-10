"""Streamlit entry point. Each file under chatqec_app/pages/ becomes a sidebar page."""

import streamlit as st

st.set_page_config(
    page_title="ChatQEC",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ChatQEC")
st.write(
    "Open-source Quantum Error Correction research assistant. " "Pick a page on the left to start."
)

st.markdown(
    """
- **Chat** — ask QEC questions with cited evidence and sandboxed Stim execution.
- **Sources** — browse the knowledge base.
- **Watchlist** — see which arxiv categories, GitHub repos, and channels we track.
- **About** — what's inside, license, how to self-host.
"""
)
