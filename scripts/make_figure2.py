"""Build the paper's Figure 2 (annotated Chat-page screenshot) from the seeded capture.

    uv run python scripts/make_figure2.py

Input:  6ac52ba22379cb285b6dc6d7/figures/screens_final/computation_answer.png, written by
        scripts/capture_figure2.sh (prompt decoder_compare_color_seeded in
        eval/ui_acceptance_prompts.json; Muse-Glimmer, 2000 shots, seed 42).
Output: 6ac52ba22379cb285b6dc6d7/figures/chat_computation.png

Two columns: left, the answer excerpt (1) and the citation cards (2); right, the tool input (3)
and the tool's result table (4). The crop rows are fixed to this capture: re-check them by eye
after a new capture, since the answer text (and so its layout) can change between runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "6ac52ba22379cb285b6dc6d7/figures"
SRC = FIG / "screens_final/computation_answer.png"
OUT = FIG / "chat_computation.png"

# (top, bottom) pixel rows of each excerpt in the 1408 px wide capture
ANSWER_HEAD = (0, 660)  # setup line and the three decoder results
ANSWER_TIE = (1890, 2220)  # the overlapping-interval (statistical tie) statement
CARDS = (2225, 2560)
TOOL_INPUT = (2570, 3260)
TOOL_TABLE = (3270, 3630)

GAP, PAD, BADGE = 36, 24, 64
ORANGE, WHITE, GREY = (234, 88, 12), (255, 255, 255), (150, 150, 150)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _crop(im: Image.Image, rows: tuple[int, int]) -> Image.Image:
    return im.crop((0, rows[0], im.width, rows[1]))


def _ellipsis(width: int) -> Image.Image:
    """A thin row with a centred ellipsis, marking text omitted between two excerpts."""
    strip = Image.new("RGB", (width, 60), WHITE)
    d = ImageDraw.Draw(strip)
    for i in range(3):
        x = width // 2 - 30 + 30 * i
        d.ellipse((x - 6, 24, x + 6, 36), fill=GREY)
    return strip


def _stack(parts: list[Image.Image]) -> Image.Image:
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts)), WHITE)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def _badge(canvas: Image.Image, n: int, xy: tuple[int, int], font: ImageFont.FreeTypeFont) -> None:
    d = ImageDraw.Draw(canvas)
    x, y = xy
    d.ellipse((x, y, x + BADGE, y + BADGE), fill=ORANGE)
    d.text((x + BADGE / 2, y + BADGE / 2), str(n), fill=WHITE, font=font, anchor="mm")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    im = Image.open(SRC).convert("RGB")
    font = ImageFont.truetype(FONT, 40)
    answer = _stack([_crop(im, ANSWER_HEAD), _ellipsis(im.width), _crop(im, ANSWER_TIE)])
    cards = _crop(im, CARDS)
    tool_in, table = _crop(im, TOOL_INPUT), _crop(im, TOOL_TABLE)

    left_h = answer.height + GAP + cards.height
    right_h = tool_in.height + GAP + table.height
    w = PAD + BADGE + im.width + GAP + BADGE + im.width + PAD
    canvas = Image.new("RGB", (w, max(left_h, right_h) + 2 * PAD), WHITE)
    lx, rx = PAD + BADGE, PAD + BADGE + im.width + GAP + BADGE
    placed = [
        (1, answer, lx, PAD),
        (2, cards, lx, PAD + answer.height + GAP),
        (3, tool_in, rx, PAD),
        (4, table, rx, PAD + tool_in.height + GAP),
    ]
    for n, img, x, y in placed:
        canvas.paste(img, (x, y))
        _badge(canvas, n, (x - BADGE - 6, y + 12), font)
    canvas.save(OUT, dpi=(300, 300))
    logger.info("wrote %s (%dx%d)", OUT.relative_to(REPO), *canvas.size)


if __name__ == "__main__":
    main()
