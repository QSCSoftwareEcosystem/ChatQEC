"""Render a looser take on the ChatQEC logo brief in
docs/images/chatqec_logo_prompt.md: a single bubble (navy outline, solid
cyan-green tail) with a loose diagonal node chain inside instead of a strict
grid -- one error node, one verified node right after it in the chain --
next to the "ChatQEC" wordmark.

Outputs SVG (editable vector), PDF and a 300-dpi PNG into docs/images/ as
chatqec_logo_concept_d.*  (does not touch the earlier chatqec_logo* files).

    uv run python scripts/make_logo_concept_d.py
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
CYAN = "#14b892"
ORANGE = "#e8571f"
WHITE = "#ffffff"

BUB_X, BUB_Y, BUB_W, BUB_H = 30.0, 50.0, 160.0, 120.0


def bubble(ax) -> None:
    tail = Polygon(
        [(BUB_X + 22, BUB_Y + 2), (BUB_X + 48, BUB_Y + 2), (BUB_X + 16, BUB_Y - 32)],
        closed=True,
        facecolor=CYAN,
        edgecolor=CYAN,
        linewidth=0,
        zorder=2,
    )
    ax.add_patch(tail)
    ax.add_patch(
        FancyBboxPatch(
            (BUB_X, BUB_Y),
            BUB_W,
            BUB_H,
            boxstyle="round,pad=0,rounding_size=24",
            linewidth=6.0,
            edgecolor=NAVY,
            facecolor=WHITE,
            zorder=3,
        )
    )
    ax.plot([BUB_X + 22, BUB_X + 48], [BUB_Y + 2, BUB_Y + 2], color=WHITE, lw=7, zorder=3.5)


def glyph(ax, cx: float, cy: float, kind: str) -> None:
    if kind == "x":
        d = 3.6
        ax.add_line(
            Line2D(
                [cx - d, cx + d],
                [cy - d, cy + d],
                color=WHITE,
                lw=2.6,
                solid_capstyle="round",
                zorder=7,
            )
        )
        ax.add_line(
            Line2D(
                [cx - d, cx + d],
                [cy + d, cy - d],
                color=WHITE,
                lw=2.6,
                solid_capstyle="round",
                zorder=7,
            )
        )
    else:
        pts = [(cx - 5.0, cy - 0.3), (cx - 1.3, cy - 4.4), (cx + 5.6, cy + 4.6)]
        ax.add_line(
            Line2D(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=WHITE,
                lw=2.6,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=7,
            )
        )


def lattice(ax) -> None:
    """A loose 4-node diagonal chain: a conversation moving forward, with
    one error caught mid-chain and corrected at the very next node."""
    a = (58.0, 95.0)
    b = (100.0, 138.0)  # error
    c = (140.0, 103.0)  # verified, right after b
    d = (160.0, 148.0)

    for p, q in ((a, b), (b, c), (c, d)):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=NAVY, lw=2.8, zorder=4, solid_capstyle="round")

    ax.add_patch(Circle(a, 7.0, facecolor=NAVY, edgecolor=WHITE, linewidth=1.6, zorder=5))
    ax.add_patch(Circle(d, 7.0, facecolor=NAVY, edgecolor=WHITE, linewidth=1.6, zorder=5))

    ax.add_patch(Circle(b, 11.5, facecolor=ORANGE, edgecolor=WHITE, linewidth=2.0, zorder=6))
    glyph(ax, *b, "x")

    ax.add_patch(Circle(c, 11.5, facecolor=CYAN, edgecolor=WHITE, linewidth=2.0, zorder=6))
    glyph(ax, *c, "check")


def wordmark(fig, ax) -> None:
    tx = BUB_X + BUB_W + 50
    chat = ax.text(
        tx,
        H / 2,
        "Chat",
        color=NAVY,
        fontsize=58,
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
        color=CYAN,
        fontsize=58,
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

    bubble(ax)
    lattice(ax)
    wordmark(fig, ax)

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_logo_concept_d.{ext}", **kw)
        print("wrote", out / f"chatqec_logo_concept_d.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
