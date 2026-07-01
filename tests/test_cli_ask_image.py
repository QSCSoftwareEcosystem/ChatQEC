from pathlib import Path

from PIL import Image

from chatqec.cli import _read_image_files


def test_read_image_files(tmp_path: Path):
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p)
    imgs = _read_image_files((str(p),))
    assert len(imgs) == 1
    assert imgs[0].media_type == "image/png"
    assert imgs[0].filename == "a.png"


def test_read_image_files_empty():
    assert _read_image_files(()) == []
