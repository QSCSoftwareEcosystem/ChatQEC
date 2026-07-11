from pathlib import Path
from PIL import Image
import numpy as np
from chatqec.ingestion.figures import is_probable_figure, annotate_figures


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


class _FakeDescriber:
    def __init__(self, text="a distance-3 surface code lattice"):
        self.text = text
        self.calls = []

    def describe(self, image_path: Path) -> str:
        self.calls.append(Path(image_path))
        return self.text


def test_annotate_injects_marked_description(tmp_path):
    _png(tmp_path, "f.png", (640, 480))
    md = "Intro ![](f.png) end"
    out, figs = annotate_figures(md, describer=_FakeDescriber(), base_dir=tmp_path)
    assert "![](f.png)" in out
    assert "> **Figure (auto-described):** a distance-3 surface code lattice" in out
    assert figs == ["f.png"]


def test_annotate_skips_decorative(tmp_path):
    _png(tmp_path, "logo.png", (16, 16))
    md = "![](logo.png)"
    fake = _FakeDescriber()
    out, figs = annotate_figures(md, describer=fake, base_dir=tmp_path)
    assert out == md and figs == [] and fake.calls == []


def test_annotate_survives_describer_failure(tmp_path):
    _png(tmp_path, "f.png", (640, 480))

    class Boom:
        def describe(self, image_path):
            raise RuntimeError("model down")

    md = "![](f.png)"
    out, figs = annotate_figures(md, describer=Boom(), base_dir=tmp_path)
    assert out == md and figs == []
