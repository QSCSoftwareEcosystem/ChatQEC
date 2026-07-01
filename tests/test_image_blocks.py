import base64

from chatqec.api_types import ImageAttachment
from chatqec.conversation.image_blocks import (
    anthropic_image_block,
    gemini_image_part,
    openai_image_part,
)

IMG = ImageAttachment(data=b"\x89PNG\r\n", media_type="image/png", filename="a.png")
B64 = base64.b64encode(IMG.data).decode()


def test_anthropic_block():
    b = anthropic_image_block(IMG)
    assert b["type"] == "image"
    assert b["source"] == {"type": "base64", "media_type": "image/png", "data": B64}


def test_openai_part():
    p = openai_image_part(IMG)
    assert p["type"] == "image_url"
    assert p["image_url"]["url"] == f"data:image/png;base64,{B64}"


def test_gemini_part():
    p = gemini_image_part(IMG)
    assert p["inline_data"] == {"mime_type": "image/png", "data": B64}
