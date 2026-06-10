"""Renders a ToolCall's output: Stim diagrams as SVG, sample arrays as tables."""

import streamlit as st

from chatqec.api import ToolCall

_MAX_INLINE_CHARS = 2000


def render(call: ToolCall) -> None:
    st.markdown(f"**Tool:** `{call.name}`")
    st.code(str(call.input), language="json")
    out = call.output
    # The output is a stringified dict from ChatQEC; try to render structured pieces
    if call.name == "stim_diagram" and "<svg" in out:
        start = out.find("<svg")
        end = out.rfind("</svg>") + len("</svg>")
        st.components.v1.html(out[start:end], height=300)
    else:
        st.code(out[:_MAX_INLINE_CHARS], language="json")
        if len(out) > _MAX_INLINE_CHARS:
            st.caption(f"(truncated; {len(out)} chars)")
