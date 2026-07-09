import io

import pytest
from PIL import Image

from chatqec.api_types import ImageAttachment
from chatqec.ingestion.images import ALLOWED_MEDIA_TYPES, ImageError, prepare_images


def _png(w=10, h=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_prepare_returns_attachments():
    out = prepare_images([(_png(), "image/png", "a.png")])
    assert len(out) == 1
    assert isinstance(out[0], ImageAttachment)
    assert out[0].media_type == "image/png"
    assert out[0].filename == "a.png"
    assert out[0].data


def test_rejects_disallowed_type():
    with pytest.raises(ImageError):
        prepare_images([(b"gif89a", "image/gif", "x.gif")])


def test_rejects_too_many():
    imgs = [(_png(), "image/png", f"{i}.png") for i in range(4)]
    with pytest.raises(ImageError):
        prepare_images(imgs, max_images=3)


def test_downscales_large_image():
    big = _png(4000, 100)
    out = prepare_images([(big, "image/png", "big.png")], max_edge=1568)
    w, h = Image.open(io.BytesIO(out[0].data)).size
    assert max(w, h) <= 1568


def test_png_media_type_in_allowed():
    assert "image/png" in ALLOWED_MEDIA_TYPES
