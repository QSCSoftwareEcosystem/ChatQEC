"""Renders a ToolCall's output: Stim diagrams as SVG, sample arrays as tables."""

import ast
import json
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


def _deltakit_rows(data: dict) -> list[dict]:
    """Metric rows for a deltakit_memory result."""
    fields = (
        ("logical error rate", "logical_error_rate"),
        ("std error", "std_error"),
        ("shots", "shots"),
        ("failures", "num_failures"),
        ("code", "code_family"),
        ("distance", "distance"),
        ("rounds", "rounds"),
        ("qubits", "num_qubits"),
        ("detectors", "num_detectors"),
        ("decoder", "decoder"),
        ("noise", "noise_model"),
    )
    return [{"metric": label, "value": data.get(key)} for label, key in fields]


def _ft_primitive_rows(data: dict) -> list[dict]:
    """Metric rows for an ft_primitive_simulate result (FTPrimitiveBench)."""
    rounds = data.get("rounds") or {}
    rounds_text = ", ".join(f"{k.removesuffix('_rounds')} {v}" for k, v in rounds.items())
    rows = [
        ("logical error rate", data.get("logical_error_rate")),
        ("95% interval", f"[{data.get('ci95_low')}, {data.get('ci95_high')}]"),
        ("shots", data.get("shots")),
        ("failures", data.get("num_failures")),
        ("primitive", data.get("primitive")),
        ("patch (X x Z distance)", f"{data.get('x_distance')} x {data.get('z_distance')}"),
        ("rounds", rounds_text),
        ("measurement basis", data.get("meas_basis")),
        ("noise", data.get("noise_model")),
        ("decoder", data.get("decoder")),
        ("qubits", data.get("num_qubits")),
        ("detectors", data.get("num_detectors")),
    ]
    return [{"metric": label, "value": value} for label, value in rows]


def _guppy_rows(data: dict) -> list[dict]:
    """Outcome rows for a guppy_run result, most frequent first."""
    registers = ",".join(data.get("registers", []))
    shots = data.get("shots") or sum(data.get("counts", {}).values())
    return [
        {
            "outcome": outcome,
            "registers": registers,
            "shots": n,
            "fraction": round(n / shots, 4) if shots else None,
        }
        for outcome, n in sorted(data.get("counts", {}).items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _code_params_rows(data: dict) -> list[dict]:
    """Literature-family rows of a code_params result: the code, its source, and where in it."""
    return [
        {
            "code": f"[[{r['n']}, {r['k']}, {r['d']}]]",
            "source": r.get("source"),
            "where": r.get("locator"),
        }
        for r in data.get("rows", [])
    ]


_ROW_BUILDERS = {
    "decoder_compare": _decoder_rows,
    "ibm_add_pauli_checks": _pauli_overhead_rows,
    "ibm_ft_compile": _ft_compile_rows,
    "deltakit_memory": _deltakit_rows,
    "ft_primitive_simulate": _ft_primitive_rows,
    "guppy_run": _guppy_rows,
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
        if data.get("stderr_tail"):
            st.code(data["stderr_tail"], language="text")
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
    if name == "code_params" and data.get("rows"):
        st.dataframe(_code_params_rows(data))
        if data.get("notes"):
            st.caption(data["notes"])
        return True
    if name in ("pymatching_decode", "code_params"):
        st.json(data)
        return True
    return False


def render(call: ToolCall) -> None:
    st.markdown(f"**Tool:** `{call.name}`")
    try:  # pretty JSON, wrapped, so long inputs (decoder lists) are not cut off
        shown = json.dumps(call.input, indent=1, default=str)
    except (TypeError, ValueError):
        shown = str(call.input)
    st.code(shown, language="json", wrap_lines=True)
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
