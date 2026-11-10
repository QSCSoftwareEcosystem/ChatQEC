"""Build the appendix figures: frontier chat replies to the Figure 2 prompt, as excerpts of their PDFs.

    uv run python scripts/make_appendix_frontier_figure.py

Inputs (browser prints saved 2026-11-10 by the authors; first reply to the same prompt as Figure 2):
    eval/compare/transcripts/chatgpt_decoder_compare_20261110.pdf  (chatgpt.com)
    eval/compare/transcripts/claude_decoder_compare_20261110.pdf   (claude.ai)
Outputs: 6ac52ba22379cb285b6dc6d7/figures/{chatgpt,claude}_computation.png

Excerpts (page, top, bottom) are in PDF points at 72 dpi and are fixed to these prints.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "eval/compare/transcripts"
FIG = REPO / "6ac52ba22379cb285b6dc6d7/figures"
DPI = 220
# name -> (source pdf, [(page, top, bottom)], (left, right)) in points
PANELS = {
    "chatgpt": (
        "chatgpt_decoder_compare_20261110.pdf",
        [(1, 55, 135), (2, 438, 738), (3, 700, 736)],
        (40, 572),
    ),
    "claude": (
        "claude_decoder_compare_20261110.pdf",
        [(1, 92, 584)],  # prompt, run summary, results table, the statistical-tie bullet
        (40, 590),
    ),
}
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


def build(name: str) -> Path:
    src, excerpts, (left, right) = PANELS[name]
    s = DPI / 72
    parts: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (page, top, bottom) in enumerate(excerpts):
            im = _page(TRANSCRIPTS / src, page, Path(tmp))
            crop = im.crop((int(left * s), int(top * s), int(right * s), int(bottom * s)))
            if i:
                parts.append(_ellipsis(crop.width))
            parts.append(crop)
    out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts)), WHITE)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    path = FIG / f"{name}_computation.png"
    out.save(path, dpi=(300, 300))
    logger.info("wrote %s (%dx%d)", path.relative_to(REPO), *out.size)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in PANELS:
        build(name)


if __name__ == "__main__":
    main()
