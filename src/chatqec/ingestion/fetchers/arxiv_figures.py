"""Harvest figures from an arXiv paper's PDF for the hybrid ingestion path.

The LaTeX route gives the best text/equation fidelity but drops figures. To keep
that text *and* gain figures, this module downloads the paper's PDF, runs Marker
purely to persist the figure images (its extracted text is discarded — the LaTeX
chunks are authoritative), then describes each real figure with the local vision
model and returns dedicated figure chunks to append to the source.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import requests

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.fetchers.pdf_marker import MarkerUnavailableError, run_marker_on_pdf
from chatqec.ingestion.figures import FigureDescriber, figures_to_chunks

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _safe(source_id: str) -> str:
    return source_id.replace(":", "_").replace("/", "_").replace("@", "_")


def fetch_arxiv_figure_chunks(
    arxiv_id: str,
    *,
    source_id: str,
    title: str,
    describer: FigureDescriber,
    start_index: int,
    images_root: Path = Path("data/figures"),
) -> list[Chunk]:  # pragma: no cover - network + subprocess + GPU
    """Download the arXiv PDF, persist its figures, and return described figure chunks.

    ``source_id`` is the paper's registry id (``arxiv:<id>``); figures are saved
    under ``images_root/<safe source_id>/`` and their links resolve from repo
    root. ``start_index`` is the next free ``chunk_index`` after the LaTeX chunks.
    Returns ``[]`` if the PDF can't be fetched or Marker produces no figures —
    figure harvesting must never break the (already-succeeded) text ingestion.
    """
    bare_id = arxiv_id.split("v", maxsplit=1)[0] if "v" in arxiv_id else arxiv_id
    url = f"https://arxiv.org/pdf/{bare_id}"
    headers = {"User-Agent": "chatqec/0.1 (+https://errorcorrectionzoo.org)"}
    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    images_dir = images_root / _safe(source_id)
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{_safe(bare_id)}.pdf"
        pdf_path.write_bytes(bytes(resp.content))
        try:
            # Marker persists figures into images_dir; its text output is unused.
            run_marker_on_pdf(pdf_path, images_dir=images_dir)
        except (MarkerUnavailableError, OSError):
            return []

    if not images_dir.is_dir():
        return []
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return figures_to_chunks(
        image_files,
        describer=describer,
        source_id=source_id,
        source_type="arxiv",
        tier="preprint",
        title=title,
        start_index=start_index,
    )
