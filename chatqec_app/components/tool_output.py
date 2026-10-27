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


def _decoder_rows(data: dict) -> list[dict]:
    """Table rows for a decoder_compare result: successes ranked by error rate, failures last."""
    ok = sorted(
        (r for r in data.get("results", []) if r.get("ok")),
        key=lambda r: (r.get("logical_error_rate", 0.0), r.get("decode_seconds", 0.0)),
    )
    failed = [r for r in data.get("results", []) if not r.get("ok")]
    rows = [
        {
            "decoder": r["decoder"],
            "logical error rate": r.get("logical_error_rate"),
            "std error": r.get("std_error"),
            "shots": r.get("shots"),
            "decode s": r.get("decode_seconds"),
            "note": r.get("note", ""),
        }
        for r in ok
    ]
    rows += [
        {
            "decoder": r["decoder"],
            "logical error rate": None,
            "std error": None,
            "shots": None,
            "decode s": None,
            "note": r.get("note", ""),
        }
        for r in failed
    ]
    return rows


def _pauli_overhead_rows(data: dict) -> list[dict]:
    """Before/after rows for an ibm_add_pauli_checks result, plus the discard rate if known."""
    o = data.get("overhead") or {}
    rows = [
        {"metric": m, "before": o.get(f"{m}_before"), "after": o.get(f"{m}_after")}
        for m in ("qubits", "gates", "depth")
    ]
    if data.get("postselection_discard_rate") is not None:
        rows.append(
            {
                "metric": "postselection discard rate",
                "before": None,
                "after": data["postselection_discard_rate"],
            }
        )
    return rows


def _ft_compile_rows(data: dict) -> list[dict]:
    """Metric rows for an ibm_ft_compile result (Clifford+T counts, or PBC rotations)."""
    if data.get("mode") == "pbc":
        rows = [{"metric": "Pauli product rotations", "value": data.get("num_rotations")}]
    else:
        rows = [
            {"metric": "T count", "value": data.get("t_count")},
            {"metric": "Clifford gates", "value": data.get("clifford_count")},
        ]
    return [
        *rows,
        {"metric": "depth", "value": data.get("depth")},
        {"metric": "qubits", "value": data.get("num_qubits")},
    ]


_ROW_BUILDERS = {
    "decoder_compare": _decoder_rows,
    "ibm_add_pauli_checks": _pauli_overhead_rows,
    "ibm_ft_compile": _ft_compile_rows,
}


def _parsed(out: str) -> dict | None:
    """The tool output as a dict when it is a Python/JSON literal dict, else None."""
    try:
        data = ast.literal_eval(out)
    except (ValueError, SyntaxError):
        return None
    return data if isinstance(data, dict) else None


def _render_dict_output(name: str, data: dict) -> bool:
    """Render a structured tool result; False when this tool has no structured view."""
    if (name in _ROW_BUILDERS or name == "threshold_sweep") and not data.get("ok", True):
        st.error(data.get("error_message") or "tool returned an error")
        return True
    if name == "threshold_sweep":
        if data.get("rows"):
            st.dataframe(data["rows"])
        svg = data.get("plot_svg") or ""
        if "<svg" in svg:
            _render_svg(svg)
        return True
    if name in _ROW_BUILDERS:
        st.dataframe(_ROW_BUILDERS[name](data))
        if data.get("notes"):
            st.caption(data["notes"])
        if data.get("circuit_qasm"):
            st.code(data["circuit_qasm"], language="text")
        return True
    if name in ("pymatching_decode", "code_params"):
        st.json(data)
        return True
    return False


def render(call: ToolCall) -> None:
    st.markdown(f"**Tool:** `{call.name}`")
    st.code(str(call.input), language="json")
    out = call.output

    if call.name == "stim_diagram" and "<svg" in out:
        start = out.find("<svg")
        end = out.rfind("</svg>") + len("</svg>")
        _render_svg(out[start:end])
        return

    data = _parsed(out)
    if data is not None and _render_dict_output(call.name, data):
        return

    st.code(out[:_MAX_INLINE_CHARS], language="json")
    if len(out) > _MAX_INLINE_CHARS:
        st.caption(f"(truncated; {len(out)} chars)")
