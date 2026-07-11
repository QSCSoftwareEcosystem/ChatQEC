"""Load figure images referenced by retrieved chunks into ImageAttachments."""

from __future__ import annotations

from pathlib import Path

from chatqec.api_types import ImageAttachment
from chatqec.ingestion.images import prepare_images
from chatqec.retrieval.hybrid import RetrievedChunk

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def load_chunk_figures(
    chunks: list[RetrievedChunk],
    *,
    base_dir: Path = Path("."),
    max_images: int = 3,
) -> list[ImageAttachment]:
    """Read up to ``max_images`` figure files referenced by ``chunks``."""
    raw: list[tuple[bytes, str, str]] = []
    seen: set[str] = set()
    for c in chunks:
        figs = c.payload.get("figures", [])
        if not isinstance(figs, list):
            figs = []
        for rel in figs:
            if rel in seen:
                continue
            seen.add(rel)
            p = base_dir / rel
            media = _MEDIA.get(p.suffix.lower())
            if media and p.is_file():
                raw.append((p.read_bytes(), media, p.name))
            if len(raw) >= max_images:
                break
        if len(raw) >= max_images:
            break
    if not raw:
        return []
    return prepare_images(raw, max_images=max_images)
