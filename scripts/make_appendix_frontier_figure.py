"""Build the appendix figure: ChatGPT's answer to the Figure 2 prompt, as an excerpt of its PDF.

    uv run python scripts/make_appendix_frontier_figure.py

Input:  eval/compare/transcripts/chatgpt_decoder_compare_20261110.pdf (chatgpt.com print, saved
        2026-11-10 by the authors; first reply to the same prompt as Figure 2)
Output: 6ac52ba22379cb285b6dc6d7/figures/chatgpt_computation.png

Excerpts (page, top, bottom) are in PDF points at 72 dpi and are fixed to this print.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval/compare/transcripts/chatgpt_decoder_compare_20261110.pdf"
OUT = REPO / "6ac52ba22379cb285b6dc6d7/figures/chatgpt_computation.png"
DPI = 220
# (page, top, bottom) in points; left/right crop is the text column
EXCERPTS = [(1, 55, 135), (2, 438, 738), (3, 700, 736)]
LEFT, RIGHT = 40, 572
WHITE, GREY = (255, 255, 255), (150, 150, 150)


def _page(pdf: Path, n: int, tmp: Path) -> Image.Image:
    stem = tmp / f"p{n}"
    subprocess.run(
        ["pdftoppm", "-f", str(n), "-l", str(n), "-r", str(DPI), "-png", "-singlefile", pdf, stem],
        check=True,
    )
    return Image.open(f"{stem}.png").convert("RGB")


def _ellipsis(width: int) -> Image.Image:
    strip = Image.new("RGB", (width, 70), WHITE)
    d = ImageDraw.Draw(strip)
    for i in range(3):
        x = width // 2 - 30 + 30 * i
        d.ellipse((x - 6, 29, x + 6, 41), fill=GREY)
    return strip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = DPI / 72
    parts: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (page, top, bottom) in enumerate(EXCERPTS):
            im = _page(SRC, page, Path(tmp))
            crop = im.crop((int(LEFT * s), int(top * s), int(RIGHT * s), int(bottom * s)))
            if i:
                parts.append(_ellipsis(crop.width))
            parts.append(crop)
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts)), WHITE)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    out.save(OUT, dpi=(300, 300))
    logger.info("wrote %s (%dx%d)", OUT.relative_to(REPO), *out.size)


if __name__ == "__main__":
    main()
