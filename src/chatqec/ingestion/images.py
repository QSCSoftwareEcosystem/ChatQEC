"""Validate + normalize user-uploaded images into ImageAttachment objects.

Shared by the Streamlit chat and the CLI so guardrails can't drift. Images are
transient — this never writes to disk or the vector store.
"""

from __future__ import annotations

import io

from PIL import Image

from chatqec.api_types import ImageAttachment

ALLOWED_MEDIA_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}
_MEDIA_TO_PIL = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


class ImageError(ValueError):
    """Raised when an uploaded image violates the guardrails."""


def _downscale(data: bytes, media_type: str, max_edge: int) -> bytes:
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_edge:
        return data
    img.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, format=_MEDIA_TO_PIL[media_type])
    return buf.getvalue()


def prepare_images(
    raw: list[tuple[bytes, str, str]],
    *,
    max_images: int = 3,
    max_bytes: int = 5_000_000,
    max_edge: int = 1568,
) -> list[ImageAttachment]:
    """Validate (type/count/size) and downscale raw ``(data, media_type, name)``."""
    if len(raw) > max_images:
        raise ImageError(f"at most {max_images} images per message (got {len(raw)})")
    out: list[ImageAttachment] = []
    for data, media_type, name in raw:
        if media_type not in ALLOWED_MEDIA_TYPES:
            raise ImageError(f"unsupported image type: {media_type!r}")
        if len(data) > max_bytes:
            raise ImageError(f"{name or 'image'} exceeds {max_bytes // 1_000_000} MB")
        out.append(
            ImageAttachment(
                data=_downscale(data, media_type, max_edge),
                media_type=media_type,
                filename=name,
            )
        )
    return out
