"""Render an alternative ChatQEC logo concept: a stabilizer plaquette with a
flagged/verified center site and two external "citation" leads, next to a
bold two-color wordmark (no subtitle).

This is a candidate exploring the tone set by docs/images/chatqec_logo_prompt.md
(engineered/circuit tone, restrained two-color palette, "chatQEC" only in the
wordmark) without copying any reference logo's literal shapes.

Outputs SVG (editable vector), PDF and a 300-dpi PNG into docs/images/ as
chatqec_logo_concept_b.*  (does not touch the existing chatqec_logo.* files).

    uv run python scripts/make_logo_concept_b.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

W, H = 900.0, 220.0  # units == px at 100 dpi -> 9.0 x 2.2 in

CARD = "#f6f8fa"
RULE = "#d8dee5"
INK = "#1b1f24"
ACCENT = "#d7263d"
PAPER = "#ffffff"

ICX, ICY = 118.0, 110.0
R = 62.0  # diamond half-diagonal


def plaquette(ax) -> None:
    """A diamond stabilizer plaquette: 4 outer nodes + a flagged center,
    with two short external leads ending in connector terminals."""
    top = (ICX, ICY + R)
    right = (ICX + R, ICY)
    bottom = (ICX, ICY - R)
    left = (ICX - R, ICY)
    center = (ICX, ICY)

    # diagonals forming the plaquette cross
    for a, b in ((top, bottom), (left, right)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=3.2, zorder=2, solid_capstyle="round")

    # outline connecting the four outer nodes (the plaquette boundary)
    outline = [top, right, bottom, left, top]
    ax.plot(
        [p[0] for p in outline],
        [p[1] for p in outline],
        color=INK,
        lw=3.2,
        zorder=2,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    # external leads: short ticks off the left/right vertices ending in
    # small connector terminals (one open, one filled -- the only spot of
    # asymmetry, echoing a "grounded" citation link out of the lattice)
    lead = 30.0
    ax.plot(
        [left[0] - lead, left[0]],
        [left[1], left[1]],
        color=INK,
        lw=3.2,
        zorder=1,
        solid_capstyle="round",
    )
    ax.add_patch(
        Circle(
            (left[0] - lead, left[1]), 7.5, facecolor=PAPER, edgecolor=INK, linewidth=3.0, zorder=3
        )
    )

    ax.plot(
        [right[0], right[0] + lead],
        [right[1], right[1]],
        color=INK,
        lw=3.2,
        zorder=1,
        solid_capstyle="round",
    )
    ax.add_patch(
        Circle(
            (right[0] + lead, right[1]),
            7.5,
            facecolor=ACCENT,
            edgecolor=ACCENT,
            linewidth=0,
            zorder=3,
        )
    )

    # outer nodes
    for x, y in (top, bottom, left, right):
        ax.add_patch(Circle((x, y), 8.5, facecolor=INK, edgecolor=PAPER, linewidth=2.0, zorder=4))

    # flagged / verified center: accent disc with a checkmark
    ax.add_patch(Circle(center, 15.5, facecolor=ACCENT, edgecolor=PAPER, linewidth=2.4, zorder=5))
    check = [
        (center[0] - 7.0, center[1] - 0.5),
        (center[0] - 2.0, center[1] - 6.0),
        (center[0] + 8.0, center[1] + 7.0),
    ]
    ax.add_line(
        Line2D(
            [p[0] for p in check],
            [p[1] for p in check],
            color=PAPER,
            lw=2.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6,
        )
    )


def wordmark(fig, ax) -> None:
    tx = ICX + R + 30 + 46
    chat = ax.text(
        tx,
        H / 2,
        "chat",
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
        tx + chat_w + 4,
        H / 2,
        "QEC",
        color=ACCENT,
        fontsize=54,
        fontweight="bold",
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

    from matplotlib.patches import FancyBboxPatch

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

    plaquette(ax)
    wordmark(fig, ax)

    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"chatqec_logo_concept_b.{ext}", **kw)
        print("wrote", out / f"chatqec_logo_concept_b.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
