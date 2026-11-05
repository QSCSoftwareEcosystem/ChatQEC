"""Clean up the FigureLabs pipeline SVG for the paper.

Input:  6ac52ba22379cb285b6dc6d7/figures/ChatQEC Research Pipeline-image-7.svg (generator output, kept)
Output: 6ac52ba22379cb285b6dc6d7/figures/chatqec_pipeline_v2.{svg,pdf,png}

- white tiles instead of the teal fill, and a pastel disc behind every icon (colour by role,
  in the style of figures/chatqec_pipeline.png)
- generator typos fixed (QAppsWiki glyph, "modal", "blogade", "BP+OSO/LSO", "Paull-based"),
  and the tool-server columns re-set as clean text (part of them had been outlined into paths)

    uv run --with svgelements --with cairosvg python scripts/fix_pipeline_svg.py
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
from svgelements import Path as SvgPath

logger = logging.getLogger(__name__)
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
FIG = Path(__file__).resolve().parent.parent / "6ac52ba22379cb285b6dc6d7/figures"
SRC = FIG / "ChatQEC Research Pipeline-image-7.svg"

TEAL = "#c5e5e1"
TILE_ORDER = range(90, 102)  # draw-order indices of the 12 tile backgrounds
BLUE, GREEN, YELLOW, LAVENDER, PEACH, PINK, GREY = (
    "#cfe2f3", "#d4ead7", "#fbeec1", "#e2dcf2", "#fde0c5", "#f6d3dc", "#e6e6e6",
)  # fmt: skip
# icon name -> (centre x, centre y, half-size) measured from the icon shapes, and disc colour
ICONS = {
    "question": (385, 46, 29, BLUE), "screen": (335, 134, 23, GREEN),
    "corpus": (145, 254, 25, YELLOW), "wiki": (515, 253, 24, GREEN),
    "qapps": (148, 349, 29, BLUE), "web": (518, 350, 28, GREY),
    "pool": (227, 459, 25, LAVENDER), "llm": (291, 554, 24, LAVENDER),
    "evidence": (56, 664, 24, BLUE), "simulate": (356, 787, 26, BLUE),
    "visualize": (462, 786, 25, GREEN), "decode": (576, 786, 22, YELLOW),
    "compile": (702, 787, 25, LAVENDER), "hardware": (838, 786, 23, PINK),
    "cited": (297, 1056, 30, PEACH),
}  # fmt: skip
TEXT_FIXES = {
    "QAppsWiki(": "QAppsWiki (via MCP)",
    "Gemini or local modal supported": "Gemini or a local model also supported",
}
DROP_TEXT = {"i (via MCP)"}
TOOL_AREA = (306, 812, 908, 962)  # x0, y0, x1, y1: tool-server text region below the icons
# tool columns: centre x, title lines, body lines
TOOL_COLUMNS = [
    (356, ["Simulate"], ["Stim", "LightStim", "QuEra Tsim"]),
    (462, ["Visualize"], ["Stim diagrams", "QuEra GLCB", "circuit viewer"]),
    (576, ["Decode"], ["PyMatching", "Tesseract", "QuEra bloqade:", "MWPF, MLE, table", "BP+OSD/LSD", "Relay-BP · CUDA-Q"]),
    (702, ["Compile &", "FT primitives"], ["Qiskit Clifford+T", "and Pauli-based", "spacetime Pauli checks", "FTPrimitiveBench"]),
    (838, ["Hardware", "& SDKs"], ["IBM Quantum:", "calibration, layout,", "job preview/submit", "Deltakit", "Guppy/Selene"]),
]  # fmt: skip


def _tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


_TRANSLATE = re.compile(r"translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)")


def _bbox(el: ET.Element) -> tuple[float, float, float, float] | None:
    """Bounding box in page coordinates. The generator positions some glyph fragments with
    transform="translate(x,y) rotate(a)"; the small rotations do not matter here, the
    translation does."""
    tag = _tag(el)
    m = _TRANSLATE.search(el.get("transform", ""))
    dx, dy = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    try:
        if tag == "path":
            b = SvgPath(el.get("d")).bbox()
        elif tag == "rect":
            x, y, w, h = (float(el.get(k, 0)) for k in ("x", "y", "width", "height"))
            b = (x, y, x + w, y + h)
        elif tag in ("circle", "ellipse"):
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            rx = float(el.get("r", el.get("rx", 0)))
            ry = float(el.get("r", el.get("ry", 0)))
            b = (cx - rx, cy - ry, cx + rx, cy + ry)
        else:
            return None
    except (TypeError, ValueError):
        return None
    return None if b is None else (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _text(x: float, y: float, s: str, size: float, bold: bool = False) -> ET.Element:
    el = ET.Element(f"{{{NS}}}text", {
        "x": f"{x:.1f}", "y": f"{y:.1f}", "text-anchor": "middle", "fill": "#000000",
        "font-family": "Arial, Helvetica, sans-serif", "font-size": f"{size}px",
        "font-weight": "700" if bold else "400",
    })  # fmt: skip
    el.text = s
    return el


def _in_tool_area(box: tuple[float, float, float, float] | None) -> bool:
    x0, y0, x1, y1 = TOOL_AREA
    return box is not None and box[1] > y0 and box[3] < y1 and box[0] > x0 and box[2] < x1


def _clean(root: ET.Element, parents: dict[ET.Element, ET.Element]) -> tuple[ET.Element, int]:
    """Whiten tiles and icon interiors, drop outlined tool text, fix typos.

    Returns the last tile background (discs are inserted after it) and the number of
    outlined-text shapes removed.
    """
    order, last_tile, removed = 0, None, 0
    for el in list(root.iter()):
        tag = _tag(el)
        if tag == "text":
            content = html.unescape(el.text or "")
            in_tools = TOOL_AREA[1] < float(el.get("y", 0)) < TOOL_AREA[3]
            if content in DROP_TEXT or (in_tools and float(el.get("x", 0)) > TOOL_AREA[0]):
                parents[el].remove(el)
            elif content in TEXT_FIXES:
                el.text = TEXT_FIXES[content]
            continue
        if tag not in ("path", "rect", "circle", "ellipse"):
            continue
        # TILE_ORDER was measured over path/rect/circle only, so ellipses are not counted.
        if tag != "ellipse":
            order += 1
        if order in TILE_ORDER and tag != "ellipse":
            last_tile = el
        past_tiles = tag == "ellipse" or order >= TILE_ORDER.stop
        if past_tiles and _in_tool_area(_bbox(el)):
            parents[el].remove(el)  # text that the generator outlined into shapes
            removed += 1
        elif last_tile is el or el.get("fill") == TEAL:
            el.set("fill", "#ffffff")  # tiles, and icon interiors on the pastel discs
    if last_tile is None:
        raise RuntimeError("tile backgrounds not found; the source SVG changed")
    return last_tile, removed


def _add_discs(last_tile: ET.Element, parents: dict[ET.Element, ET.Element]) -> None:
    """Pastel discs right after the last tile: above the white tiles, below the icons."""
    holder = parents[last_tile]
    at = list(holder).index(last_tile) + 1
    for i, (cx, cy, half, colour) in enumerate(ICONS.values()):
        attrs = {"cx": str(cx), "cy": str(cy), "r": str(half + 9), "fill": colour}
        holder.insert(at + i, ET.Element(f"{{{NS}}}circle", attrs))


def _add_tool_text(root: ET.Element) -> None:
    for cx, titles, body in TOOL_COLUMNS:
        y = 832.0
        for t in titles:
            root.append(_text(cx, y, t, 16))
            y += 18
        y += 2
        for line in body:
            root.append(_text(cx, y, line, 11.5))
            y += 15


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tree = ET.parse(SRC)
    root = tree.getroot()
    parents = {c: p for p in root.iter() for c in p}
    last_tile, removed = _clean(root, parents)
    _add_discs(last_tile, parents)
    _add_tool_text(root)
    out = FIG / "chatqec_pipeline_v2.svg"
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    cairosvg.svg2pdf(url=str(out), write_to=str(out.with_suffix(".pdf")))
    cairosvg.svg2png(url=str(out), write_to=str(out.with_suffix(".png")), output_width=1856)
    logger.info("wrote %s (+ .pdf, .png); removed %d outlined-text shapes", out, removed)


if __name__ == "__main__":
    main()
