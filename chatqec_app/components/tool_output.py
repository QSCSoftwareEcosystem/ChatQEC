"""Renders a ToolCall's output: Stim diagrams as SVG, sample arrays as tables."""

import ast

import streamlit as st

from chatqec.api import ToolCall

_MAX_INLINE_CHARS = 2000


def render(call: ToolCall) -> None:
    st.markdown(f"**Tool:** `{call.name}`")
    st.code(str(call.input), language="json")
    out = call.output

    if call.name == "stim_diagram" and "<svg" in out:
        start = out.find("<svg")
        end = out.rfind("</svg>") + len("</svg>")
        st.components.v1.html(out[start:end], height=300)
        return

    if call.name == "threshold_sweep":
        try:
            data = ast.literal_eval(out)
        except (ValueError, SyntaxError):
            data = None
        if isinstance(data, dict):
            if not data.get("ok", True):
                st.error(data.get("error_message") or "tool returned an error")
                return
            if data.get("rows"):
                st.dataframe(data["rows"])
            svg = data.get("plot_svg") or ""
            if "<svg" in svg:
                st.components.v1.html(svg, height=380)
            return

    if call.name in ("pymatching_decode", "code_params"):
        try:
            data = ast.literal_eval(out)
        except (ValueError, SyntaxError):
            data = None
        if isinstance(data, dict):
            st.json(data)
            return

    st.code(out[:_MAX_INLINE_CHARS], language="json")
    if len(out) > _MAX_INLINE_CHARS:
        st.caption(f"(truncated; {len(out)} chars)")
