"""Render the current docs/images/chatqec_logo_prompt.md brief using its
"QEC logical qubit" option: a layered navy/dark-teal chat bubble containing
a small surface-code-style patch (alternating data/stabilizer sites) with
one flagged error site and one flagged verified site, next to the
"ChatQEC" wordmark.

Outputs SVG (editable vector), PDF and a 300-dpi PNG into docs/images/ as
chatqec_logo_concept_e.*  (does not touch the earlier chatqec_logo* files).

    uv run python scripts/make_logo_concept_e.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

W, H = 880.0, 240.0  # ~3.67:1

NAVY = "#12284f"
TEAL = "#0c8f76"  # darker, teal-leaning cyan-green per the updated prompt
ORANGE = "#e8571f"
WHITE = "#ffffff"

BACK_X, BACK_Y, BACK_W, BACK_H = 50.0, 118.0, 120.0, 88.0
FRONT_X, FRONT_Y, FRONT_W, FRONT_H = 22.0, 48.0, 152.0, 112.0


def back_bubble(ax) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (BACK_X, BACK_Y),
            BACK_W,
            BACK_H,
            boxstyle="round,pad=0,rounding_size=18",
            linewidth=5.5,
            edgecolor=NAVY,
            facecolor=WHITE,
            zorder=1,
        )
    )


def front_bubble(ax) -> None:
    tail = Polygon(
        [(FRONT_X + 18, FRONT_Y + 2), (FRONT_X + 42, FRONT_Y + 2), (FRONT_X + 12, FRONT_Y - 30)],
        closed=True,
        facecolor=WHITE,
        edgecolor=TEAL,
        linewidth=5.5,
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(tail)
    ax.add_patch(
        FancyBboxPatch(
            (FRONT_X, FRONT_Y),
            FRONT_W,
            FRONT_H,
            boxstyle="round,pad=0,rounding_size=20",
            linewidth=5.5,
            edgecolor=TEAL,
            facecolor=WHITE,
            zorder=3,
        )
    )
    ax.plot([FRONT_X + 18, FRONT_X + 42], [FRONT_Y + 2, FRONT_Y + 2], color=WHITE, lw=7, zorder=3.5)


def glyph(ax, cx: float, cy: float, kind: str) -> None:
    if kind == "x":
        d = 3.4
        ax.add_line(
            Line2D(
                [cx - d, cx + d],
                [cy - d, cy + d],
                color=WHITE,
                lw=2.4,
                solid_capstyle="round",
                zorder=7,
            )
        )
        ax.add_line(
            Line2D(
                [cx - d, cx + d],
                [cy + d, cy - d],
                color=WHITE,
                lw=2.4,
                solid_capstyle="round",
                zorder=7,
            )
        )
    else:
        pts = [(cx - 4.5, cy - 0.3), (cx - 1.2, cy - 4.0), (cx + 5.0, cy + 4.2)]
        ax.add_line(
            Line2D(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=WHITE,
                lw=2.4,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=7,
            )
        )


def logical_qubit_patch(ax) -> None:
    """A 3x3 surface-code-style patch: alternating navy/teal sites on a
    grid, one flagged as an error (orange X) and one flagged as verified
    (teal check) -- a small "QEC logical qubit" rendered inside the bubble."""
    cols = [54.0, 98.0, 142.0]
    rows = [140.0, 105.0, 70.0]

    for y in rows:
        ax.plot([cols[0], cols[-1]], [y, y], color=NAVY, lw=1.8, zorder=4, alpha=0.55)
    for x in cols:
        ax.plot([x, x], [rows[-1], rows[0]], color=NAVY, lw=1.8, zorder=4, alpha=0.55)

    error_site = (1, 1)  # centre
    verified_site = (2, 1)  # immediately to its right

    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            if (c, r) == error_site:
                ax.add_patch(
                    Circle((x, y), 10.5, facecolor=ORANGE, edgecolor=WHITE, linewidth=2.0, zorder=6)
                )
                glyph(ax, x, y, "x")
            elif (c, r) == verified_site:
                ax.add_patch(
                    Circle((x, y), 10.5, facecolor=TEAL, edgecolor=WHITE, linewidth=2.0, zorder=6)
                )
                glyph(ax, x, y, "check")
            else:
                color = NAVY if (r + c) % 2 == 0 else TEAL
                ax.add_patch(
                    Circle((x, y), 6.0, facecolor=color, edgecolor=WHITE, linewidth=1.4, zorder=5)
                )


def wordmark(fig, ax) -> None:
    tx = FRONT_X + FRONT_W + 44
    chat = ax.text(
        tx,
        H / 2,
        "Chat",
        color=NAVY,
        fontsize=56,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )
    fig.canvas.draw()
    chat_w = chat.get_window_extent().transformed(ax.transData.inverted()).width
    ax.text(
        tx + chat_w + 2,
        H / 2,
        "QEC",
        color=TEAL,
        fontsize=56,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )


def main() -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor=WHITE, edgecolor="none", zorder=0))

    back_bubble(ax)
    front_bubble(ax)
    logical_qubit_patch(ax)
    wordmark(fig, ax)

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_logo_concept_e.{ext}", **kw)
        print("wrote", out / f"chatqec_logo_concept_e.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
