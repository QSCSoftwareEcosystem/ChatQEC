"""Render the ChatQEC README logo: a chat bubble holding a syndrome-flagged
surface-code lattice, next to the wordmark.

Outputs SVG (editable vector), PDF and a 300-dpi PNG into docs/images/.

    uv run python scripts/make_logo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_overview_figure import BLUE, GREEN, INK, PAPER, RULE, VERM
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

W, H = 900.0, 220.0  # units == px at 100 dpi  ->  9.0 x 2.2 in

CARD = "#f6f8fa"
BUBBLE_CX, BUBBLE_CY, BUBBLE_R = 118.0, 118.0, 82.0


def bubble(ax) -> None:
    """A chat-bubble outline (rounded square + tail) framing the lattice."""
    x, y, w, h = BUBBLE_CX - BUBBLE_R, BUBBLE_CY - BUBBLE_R + 8, 2 * BUBBLE_R, 2 * BUBBLE_R
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=22",
            linewidth=5.5,
            edgecolor=BLUE,
            facecolor=PAPER,
            zorder=2,
        )
    )
    tail = Polygon(
        [(BUBBLE_CX - 30, y + 2), (BUBBLE_CX - 8, y + 2), (BUBBLE_CX - 34, y - 26)],
        closed=True,
        facecolor=PAPER,
        edgecolor=BLUE,
        linewidth=5.5,
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(tail)
    # redraw the base edge over the tail's top seam so the join looks clean
    ax.plot([BUBBLE_CX - 30, BUBBLE_CX - 8], [y + 2, y + 2], color=PAPER, lw=7, zorder=3)


_LATTICE_SIZE = 3


def lattice(ax) -> None:
    """3x3 surface-code-style dot lattice, one site flagged as a syndrome."""
    step = 34.0
    x0, y0 = BUBBLE_CX - step, BUBBLE_CY + 8
    pts = [(x0 + c * step, y0 + r * step) for r in (-1, 0, 1) for c in (-1, 0, 1)]

    for i, (px, py) in enumerate(pts):
        r, c = divmod(i, _LATTICE_SIZE)
        if c < _LATTICE_SIZE - 1:
            ax.plot([px, px + step], [py, py], color=RULE, lw=2.4, zorder=4)
        if r < _LATTICE_SIZE - 1:
            ax.plot([px, px], [py, py + step], color=RULE, lw=2.4, zorder=4)

    flagged = 4  # centre site
    for i, (px, py) in enumerate(pts):
        if i == flagged:
            ax.add_patch(
                Circle((px, py), 12.5, facecolor=VERM, edgecolor=PAPER, linewidth=2.2, zorder=6)
            )
            ax.add_patch(
                Circle(
                    (px, py),
                    19.5,
                    facecolor="none",
                    edgecolor=VERM,
                    linewidth=2.2,
                    linestyle=(0, (2, 2)),
                    zorder=6,
                )
            )
        else:
            color = BLUE if (i % 2 == 0) else GREEN
            ax.add_patch(
                Circle((px, py), 9.5, facecolor=color, edgecolor=PAPER, linewidth=1.6, zorder=5)
            )


def wordmark(fig, ax) -> None:
    tx = BUBBLE_CX + BUBBLE_R + 46
    chat = ax.text(
        tx,
        H / 2 + 4,
        "Chat",
        color=INK,
        fontsize=54,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )
    fig.canvas.draw()
    chat_w = chat.get_window_extent().transformed(ax.transData.inverted()).width
    ax.text(
        tx + chat_w + 6,
        H / 2 + 4,
        "QEC",
        color=BLUE,
        fontsize=54,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )
    ax.text(
        tx,
        H / 2 - 44,
        "quantum error correction, cited",
        color="#5b6570",
        fontsize=15,
        ha="left",
        va="center",
        zorder=5,
    )


def main() -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (2, 2),
            W - 4,
            H - 4,
            boxstyle="round,pad=0,rounding_size=18",
            linewidth=1.4,
            edgecolor=RULE,
            facecolor=CARD,
            zorder=0,
        )
    )

    bubble(ax)
    lattice(ax)
    wordmark(fig, ax)

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_logo.{ext}", **kw)
        print("wrote", out / f"chatqec_logo.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
