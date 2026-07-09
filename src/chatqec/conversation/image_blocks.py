"""Convert ImageAttachment into each provider's image content shape."""

from __future__ import annotations

import base64
from typing import Any

from chatqec.api_types import ImageAttachment


def _b64(img: ImageAttachment) -> str:
    return base64.b64encode(img.data).decode()


def anthropic_image_block(img: ImageAttachment) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": img.media_type, "data": _b64(img)},
    }


def openai_image_part(img: ImageAttachment) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{img.media_type};base64,{_b64(img)}"},
    }


def gemini_image_part(img: ImageAttachment) -> dict[str, Any]:
    return {"inline_data": {"mime_type": img.media_type, "data": _b64(img)}}
