"""Heuristics for figure images extracted from PDFs during ingestion."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

from PIL import Image

logger = logging.getLogger(__name__)


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


_IMG_REF_RE = re.compile(r"(!\[[^\]]*\]\(([^)]+)\))")


class FigureDescriber(Protocol):
    def describe(self, image_path: Path) -> str: ...


def annotate_figures(
    md_text: str,
    *,
    describer: FigureDescriber,
    base_dir: Path = Path("."),
) -> tuple[str, list[str]]:
    """Append an auto-described blockquote after each real figure link.

    Returns the annotated markdown and the list of figure paths (as they appear
    in the links) that were described. Decorative images, missing files, and
    describer failures leave the markdown unchanged for that reference.
    """
    figures: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        whole, rel = m.group(1), m.group(2)
        path = base_dir / rel
        if not is_probable_figure(path):
            return whole
        try:
            desc = describer.describe(path).strip()
        except Exception:  # never break ingestion on a model error
            logger.warning("figure describe failed for %s", rel, exc_info=True)
            return whole
        if not desc:
            return whole
        figures.append(rel)
        return f"{whole}\n\n> **Figure (auto-described):** {desc}\n"

    return _IMG_REF_RE.sub(_sub, md_text), figures
