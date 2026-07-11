"""Heuristics for figure images extracted from PDFs during ingestion."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def is_probable_figure(path: Path, *, min_bytes: int = 3000, min_edge: int = 96) -> bool:
    """True if ``path`` is a real figure (not decorative page furniture).

    Cheap two-part gate: the file must be non-trivially large on disk AND its
    smaller dimension must exceed ``min_edge`` px. Missing/unreadable files are
    rejected rather than raising.
    """
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        with Image.open(path) as im:
            w, h = im.size
    except OSError:
        return False
    return min(w, h) >= min_edge
