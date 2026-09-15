"""Render the ChatQEC one-slide overview infographic.

Outputs SVG (editable vector, drops into PowerPoint/Keynote/LaTeX), PDF and
a 300-dpi PNG into docs/images/.

    uv run python scripts/make_overview_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- palette ---
# Okabe-Ito, colourblind safe.
INK = "#1b1f24"
MUTED = "#5b6570"
RULE = "#c9d1d9"
BLUE = "#0072B2"  # pipeline
SKY = "#56B4E9"  # evidence
ORANGE = "#E69F00"  # QAppsWiki (new this week)
VERM = "#D55E00"  # hardware-aware (new this week)
GREEN = "#009E73"  # tools + answer
PAPER = "#ffffff"

W, H = 1333.0, 750.0  # figure units == px at 100 dpi (16:9)


def box(ax, x, y, w, h, *, edge, face, lw=1.6, ls="solid", z=2, radius=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            linestyle=ls,
            zorder=z,
        )
    )


def title(ax, x, y, text, *, color=INK, size=12.5, weight="bold", ha="left"):
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight, ha=ha, va="top", zorder=5)


def body(ax, x, y, text, *, color=MUTED, size=9.6, ha="left", weight="normal"):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        ha=ha,
        va="top",
        linespacing=1.45,
        fontweight=weight,
        zorder=5,
    )


def arrow(ax, p0, p1, *, color=BLUE, lw=1.8, ls="-", rad=0.0, z=3, head=11):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def chip(ax, x, y, w, h, text, *, edge, face, size=9.8, weight="bold", tc=INK):
    box(ax, x, y, w, h, edge=edge, face=face, lw=1.5)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=tc,
        zorder=5,
    )


def badge(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        fontsize=7.6,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"),
    )


def main() -> None:  # noqa: PLR0915 -- one statement per diagram element, not complexity
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ------------------------------------------------------------- header ---
    ax.text(34, 730, "ChatQEC", fontsize=30, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(
        252,
        726,
        "an evidence-grounded research assistant for",
        fontsize=15.5,
        color=BLUE,
        ha="left",
        va="top",
    )
    ax.text(252, 704, "Quantum Error Correction", fontsize=15.5, color=BLUE, ha="left", va="top")
    body(
        ax,
        762,
        726,
        "Every claim is traced to a source ID, or produced by a real\n"
        "stabilizer simulator. When the evidence is not in the corpus,\n"
        "ChatQEC refuses instead of inventing a threshold.",
        size=10.4,
    )
    ax.plot([34, W - 34], [668, 668], color=RULE, lw=1.2, zorder=1)

    # --------------------------------------------------- row A: query in ---
    ay, ah = 594, 50
    body(ax, 34, 658, "QUERY INTAKE", size=8.6, color=MUTED, weight="bold")
    chip(
        ax,
        34,
        ay,
        286,
        ah,
        '"What is the surface code threshold\nunder depolarizing noise?"',
        edge=RULE,
        face="#f6f8fa",
        size=9.0,
        weight="normal",
    )
    arrow(ax, (326, ay + ah / 2), (356, ay + ah / 2))
    chip(
        ax,
        362,
        ay,
        224,
        ah,
        "Haiku off-topic gate\npolite refusal if not QEC",
        edge=BLUE,
        face="#eaf3fa",
        size=9.0,
        weight="normal",
    )
    arrow(ax, (592, ay + ah / 2), (622, ay + ah / 2))
    chip(
        ax,
        628,
        ay,
        262,
        ah,
        "Haiku query rewriter\npronouns resolved, 2-3 variants",
        edge=BLUE,
        face="#eaf3fa",
        size=9.0,
        weight="normal",
    )

    # fan-out into the evidence + retrieval lanes
    arrow(ax, (200, ay - 6), (200, 552), color=BLUE, lw=1.5, ls=(0, (4, 3)))
    arrow(ax, (548, ay - 6), (548, 536), color=BLUE, lw=1.5, ls=(0, (4, 3)))
    body(
        ax,
        920,
        626,
        "the rewritten query fans out to\nevery evidence source below",
        size=9.0,
        color=BLUE,
        weight="bold",
    )

    # -------------------------------------------------- row B: main lanes ---
    body(ax, 34, 574, "EVIDENCE LAYER", size=8.6, color=MUTED, weight="bold")
    body(ax, 414, 574, "RETRIEVAL", size=8.6, color=MUTED, weight="bold")
    body(ax, 718, 574, "SYNTHESIS + TOOLS", size=8.6, color=MUTED, weight="bold")
    body(ax, 1060, 574, "ANSWER", size=8.6, color=MUTED, weight="bold")

    ex, ew = 34, 344

    box(ax, ex, 466, ew, 78, edge=SKY, face="#eef8fd")
    title(ax, ex + 14, 534, "Curated corpus  (Qdrant, hybrid)", size=10.2, color=INK)
    body(
        ax,
        ex + 14,
        516,
        "2,935 chunks / 123 sources: arXiv full text, book\n"
        "chapters, vendor and lab pages, pinned GitHub docs,\n"
        "vision-described figures",
        size=8.2,
    )

    box(ax, ex, 378, ew, 76, edge=SKY, face="#eef8fd")
    title(ax, ex + 14, 444, "Canonical QEC wiki", size=10.2, color=INK)
    body(
        ax,
        ex + 14,
        426,
        "60 hand-reviewed pages, force-injected so definitional\n"
        "questions always get the right page first",
        size=8.2,
    )

    box(ax, ex, 268, ew, 98, edge=SKY, face="#eef8fd")
    title(ax, ex + 14, 356, "QAppsWiki  (over MCP)", size=10.2, color=INK)
    body(
        ax,
        ex + 14,
        336,
        "691 concept pages / ~3,778 sections, of which 649 are\n"
        "Error Correction Zoo code pages and 42 are QEC/QEM\n"
        "self-hosted BM25 search_pages over a local clone\n"
        "tiers wiki_curated / wiki_unverified from frontmatter",
        size=8.2,
    )

    box(ax, ex, 214, ew, 42, edge=RULE, face="#f6f8fa", ls=(0, (4, 3)))
    title(ax, ex + 14, 247, "Web fallback (Tavily), off by default", size=9.4, color=MUTED)
    body(ax, ex + 14, 231, "fires only when the best rerank score < 0.4; lowest tier", size=8.2)

    # retriever
    rx, rw = 414, 268
    box(ax, rx, 300, rw, 228, edge=BLUE, face="#eaf3fa", lw=2.2)
    title(ax, rx + 16, 518, "Hybrid retriever", size=11.5, color=BLUE)
    body(
        ax,
        rx + 16,
        496,
        "0.50 dense  +  0.30 BM25\n+ 0.15 authority tier  + 0.05 recency",
        size=8.8,
        color=INK,
    )
    ax.plot([rx + 16, rx + rw - 16], [450, 450], color="#b9d6ea", lw=1.1, zorder=3)
    body(ax, rx + 16, 440, "BGE cross-encoder rerank", size=9.6, color=INK, weight="bold")
    body(
        ax,
        rx + 16,
        420,
        "corpus chunks and QAppsWiki sections\n"
        "land in ONE candidate pool, scored on\nthe same scale, then top-K",
        size=8.4,
    )
    ax.plot([rx + 16, rx + rw - 16], [368, 368], color="#b9d6ea", lw=1.1, zorder=3)
    body(
        ax,
        rx + 16,
        358,
        "the canonical pages bypass this and\n"
        "are injected directly, so definitional\nquestions cannot be outranked",
        size=8.4,
    )

    for y0, y1 in ((505, 500), (416, 458), (317, 412), (235, 356)):
        arrow(ax, (ex + ew + 5, y0), (rx - 5, y1), color=SKY, lw=1.8, rad=0.05)

    # synthesizer
    sx, sw = 718, 306
    box(ax, sx, 420, sw, 108, edge=GREEN, face="#e8f6f1", lw=2.2)
    title(ax, sx + 16, 518, "Large language model (Claude)", size=10.6, color="#00654a")
    body(
        ax,
        sx + 16,
        496,
        "native tool-use loop: decides by itself when\n"
        "a question needs a computation rather than a\n"
        "quotation, and swaps providers automatically\nif one API is unavailable",
        size=8.4,
    )
    arrow(ax, (rx + rw + 5, 460), (sx - 5, 470), color=BLUE, lw=2.4)

    # tools
    box(ax, sx, 214, sw, 190, edge=GREEN, face="#f2fbf8")
    title(ax, sx + 14, 396, "MCP tool servers  (sandboxed)", size=10.2, color="#00654a")
    ty = 374
    for name, detail in (
        ("Stim", "circuit simulation, diagrams,"),
        ("", "code parameters and distance"),
        ("PyMatching", "MWPM decoding, threshold sweeps"),
        ("LightStim", "toric, BB, color-code circuits"),
        ("Tsim", "non-Clifford gates"),
        ("GLCB", "shareable circuit links"),
        ("IBM Quantum", "live calibration, qubit-layout advice"),
    ):
        if name:
            ax.text(
                sx + 14,
                ty,
                name,
                fontsize=7.8,
                fontweight="bold",
                color="#00654a",
                ha="left",
                va="top",
                zorder=5,
            )
        ax.text(sx + 106, ty, detail, fontsize=7.8, color=MUTED, ha="left", va="top", zorder=5)
        ty -= 15
    body(
        ax,
        sx + 14,
        ty - 6,
        "subprocess isolation, memory and wall-clock\nceilings; every number comes from the tool",
        size=7.8,
    )
    arrow(ax, (sx + 96, 418), (sx + 96, 406), color=GREEN, lw=1.6)
    arrow(ax, (sx + 216, 406), (sx + 216, 418), color=GREEN, lw=1.6)

    # answer
    nx, nw = 1060, 239
    box(ax, nx, 214, nw, 314, edge=GREEN, face="#e8f6f1", lw=2.2)
    title(ax, nx + 15, 518, "Cited answer", size=12.2, color="#00654a")
    for y, txt in (
        (490, "inline citations to source IDs\nthe reader can click and verify"),
        (438, "reported numbers come from\nthe tool's actual output"),
        (386, "circuit SVGs, threshold plots\nand ingested figures inline"),
        (334, "device-aware layout advice\nwhen a backend is named"),
        (282, "honest refusal when the corpus\ndoes not cover the question"),
    ):
        body(ax, nx + 15, y, txt, size=8.6, color=INK)
    arrow(ax, (sx + sw + 5, 478), (nx - 5, 478), color=GREEN, lw=2.4)

    # ------------------------------------------------- row C: what backs it ---
    cw = (W - 68 - 24) / 3
    for i, (h, d) in enumerate(
        (
            ("Guards", "per-IP rate limit  ·  daily USD budget  ·  hCaptcha"),
            (
                "Eval",
                "50-pair Q&A set  ·  recall@K  ·  citation accuracy\n" "LLM-judge faithfulness",
            ),
            (
                "Interfaces",
                "Python library  ·  chatqec CLI (ask / chat)\n" "Streamlit app (4 pages)",
            ),
        )
    ):
        cx = 34 + i * (cw + 12)
        box(ax, cx, 96, cw, 78, edge=RULE, face="#f6f8fa", lw=1.4)
        title(ax, cx + 14, 164, h, size=9.6, color=INK)
        body(ax, cx + 14, 146, d, size=8.4)

    # ------------------------------------------------------------- footer ---
    ax.plot([34, W - 34], [48, 48], color=RULE, lw=1.0, zorder=1)
    body(ax, 34, 40, "416 tests, 90% line coverage", size=8.6)
    body(ax, W - 34, 40, "v0.1.0  ·  github.com/QSCSoftwareThrust/ChatQEC", size=8.6, ha="right")

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_overview.{ext}", facecolor=PAPER, **kw)
        print("wrote", out / f"chatqec_overview.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
