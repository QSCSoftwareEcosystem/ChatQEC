from PIL import Image
import numpy as np
from chatqec.ingestion.figures import is_probable_figure


def _png(tmp_path, name, size):
    p = tmp_path / name
    # Create image with random pixel data to ensure file size >= 3000 bytes
    # (solid-color images compress too well with PNG)
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(p, format="PNG")
    return p


def test_rejects_tiny_dimension_image(tmp_path):
    assert is_probable_figure(_png(tmp_path, "icon.png", (32, 32))) is False


def test_accepts_reasonable_figure(tmp_path):
    assert is_probable_figure(_png(tmp_path, "fig.png", (640, 480))) is True


def test_rejects_missing_file(tmp_path):
    assert is_probable_figure(tmp_path / "nope.png") is False
