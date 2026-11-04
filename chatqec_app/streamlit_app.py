"""Streamlit entry point. Each file under chatqec_app/pages/ becomes a sidebar page."""

import streamlit as st

from chatqec_app.utils import ICON_PATH

st.set_page_config(
    page_title="ChatQEC",
    page_icon=str(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chat is the page users actually want on load; land there instead of this
# intro screen. st.switch_page runs once per script execution, so this fires
# immediately on every visit to the bare app URL.
st.switch_page("pages/1_Chat.py")
