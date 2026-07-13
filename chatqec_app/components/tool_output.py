"""Renders a ToolCall's output: Stim diagrams as SVG, sample arrays as tables."""

import ast
import re

import streamlit as st

from chatqec.api import ToolCall

_MAX_INLINE_CHARS = 2000


def _svg_render_height(svg: str, *, width: int = 700, cap: int = 1200, default: int = 420) -> int:
    """Pixel height for the SVG iframe so the whole figure shows at container width.

    A hard-coded height clips tall diagrams (a Stim timeline is ``viewBox="0 0 1120
    672"``). Derive the height from the SVG's aspect ratio (viewBox, else a height
    attr), scaled to the assumed container ``width`` and capped so a very tall SVG
    stays scrollable rather than unbounded.
    """
    m = re.search(r'viewBox\s*=\s*"[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"', svg)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        if w > 0:
            return max(200, min(cap, int(width * h / w) + 20))
    mh = re.search(r'\bheight\s*=\s*"([\d.]+)', svg)
    if mh:
        return max(200, min(cap, int(float(mh.group(1))) + 20))
    return default


def _render_svg(svg: str) -> None:
    """Render an SVG responsively: full container width, height from its aspect ratio."""
    html = f"<style>svg{{width:100%;height:auto;display:block;}}</style>{svg}"
    st.components.v1.html(html, height=_svg_render_height(svg), scrolling=True)


def render(call: ToolCall) -> None:
    st.markdown(f"**Tool:** `{call.name}`")
    st.code(str(call.input), language="json")
    out = call.output

    if call.name == "stim_diagram" and "<svg" in out:
        start = out.find("<svg")
        end = out.rfind("</svg>") + len("</svg>")
        _render_svg(out[start:end])
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
                _render_svg(svg)
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
