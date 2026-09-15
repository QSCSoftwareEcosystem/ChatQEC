"""Render the vertical ChatQEC panel that fills the blank left half of the
DOE/QSC highlight slide.

Sized to ~5.3 x 5.5 in so it drops in at 100% with no rescaling. Deliberately
avoids restating the slide's own prose: it carries the flow, the counts, and
the two things the slide text does not mention (QAppsWiki, hardware-aware).

    uv run python scripts/make_slide_panel_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_overview_figure import (
    BLUE,
    GREEN,
    INK,
    MUTED,
    PAPER,
    RULE,
    SKY,
    arrow,
    body,
    box,
    chip,
    title,
)

W, H = 528.0, 547.0  # units == px at 100 dpi  ->  5.28 x 5.47 in


def main() -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    L, R = 16.0, 512.0
    CX = (L + R) / 2

    # ---- A. the question ---------------------------------------------------
    chip(
        ax,
        130,
        515,
        268,
        26,
        "a researcher's question",
        edge=RULE,
        face="#f6f8fa",
        size=8.8,
        weight="normal",
    )
    arrow(ax, (CX, 513), (CX, 507), color=BLUE, lw=1.6, head=10)

    # ---- B. screen + rewrite ----------------------------------------------
    chip(
        ax,
        108,
        477,
        312,
        26,
        "screened, then rewritten into search variants",
        edge=BLUE,
        face="#eaf3fa",
        size=8.8,
        weight="normal",
    )
    arrow(ax, (CX, 475), (CX, 466), color=BLUE, lw=1.6, head=10)

    # ---- C. the evidence it searches at once -------------------------------
    box(ax, L, 324, R - L, 140, edge=SKY, face="#fbfdff", lw=1.4)
    rows = [
        (427, "Qdrant store", "2,935 chunks across 123 sources"),
        (393, "60 core pages", "hand-reviewed, injected ahead of ranking"),
        (359, "QAppsWiki", "691 pages, 649 of them Error Correction Zoo"),
        (325, "web search", "only if the best score falls below 0.4"),
    ]
    for y, name, detail in rows:
        ax.add_patch(plt.Rectangle((L + 8, y + 3), 4, 24, color=SKY, zorder=4))
        ax.text(
            L + 20,
            y + 15,
            name,
            fontsize=8.6,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
            zorder=5,
        )
        ax.text(
            L + 152, y + 15, detail, fontsize=8.0, color=MUTED, ha="left", va="center", zorder=5
        )
    for hy in (425, 391, 357):
        ax.plot([L + 8, R - 8], [hy, hy], color="#e4ebf1", lw=0.9, zorder=3)
    arrow(ax, (CX, 322), (CX, 314), color=BLUE, lw=1.6, head=10)

    # ---- D. one pool -------------------------------------------------------
    box(ax, L, 270, R - L, 42, edge=BLUE, face="#eaf3fa", lw=2.0)
    title(ax, L + 14, 305, "one ranked pool", size=9.4, color=BLUE)
    body(ax, L + 14, 290, "outside pages compete head-to-head with local text", size=8.0)
    arrow(ax, (CX, 268), (CX, 260), color=BLUE, lw=1.6, head=10)

    # ---- E. writing, and the tools it can reach ----------------------------
    box(ax, L, 104, 214, 156, edge=GREEN, face="#e8f6f1", lw=2.0)
    title(ax, L + 12, 253, "a large language model", size=9.4, color="#00654a")
    body(
        ax,
        L + 12,
        238,
        "(Claude) drafts the answer\nand decides for itself when\n"
        "a question needs a computation\nrather than a quotation",
        size=7.8,
    )
    body(ax, L + 12, 166, "it swaps providers if\none API is unavailable", size=7.6)

    box(ax, 244, 104, 268, 156, edge=GREEN, face="#f2fbf8", lw=1.6)
    title(ax, 256, 253, "tools it can reach", size=9.4, color="#00654a")
    ty = 237
    for name, detail in (
        ("Stim", "circuit simulation, diagrams,"),
        ("", "code parameters and distance"),
        ("PyMatching", "MWPM decoding and"),
        ("", "threshold sweeps"),
        ("LightStim", "toric, BB, color codes"),
        ("Tsim", "non-Clifford gates"),
        ("GLCB", "shareable circuit links"),
        ("IBM Quantum", "live calibration and"),
        ("", "qubit-layout advice"),
    ):
        if name:
            ax.text(
                256,
                ty,
                name,
                fontsize=7.5,
                fontweight="bold",
                color="#00654a",
                ha="left",
                va="top",
                zorder=5,
            )
        ax.text(344, ty, detail, fontsize=7.5, color=MUTED, ha="left", va="top", zorder=5)
        ty -= 13.0
    body(ax, 256, ty - 4, "sandboxed: memory and wall-clock limits", size=7.2)

    arrow(ax, (230, 200), (242, 200), color=GREEN, lw=1.5, head=9)
    arrow(ax, (242, 182), (230, 182), color=GREEN, lw=1.5, head=9)
    arrow(ax, (123, 102), (123, 92), color=GREEN, lw=1.6, head=10)

    # ---- F. what comes out -------------------------------------------------
    box(ax, L, 18, R - L, 72, edge=GREEN, face="#e8f6f1", lw=2.2)
    title(ax, L + 14, 82, "the answer", size=9.8, color="#00654a")
    body(
        ax,
        L + 14,
        66,
        "every number comes from a tool run, never a guess\n"
        "it refuses outright when the evidence is missing\n"
        "shipped as a Python library, a CLI and a web app",
        size=8.0,
        color=INK,
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_slide_panel.{ext}", facecolor=PAPER, **kw)
        print("wrote", out / f"chatqec_slide_panel.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
