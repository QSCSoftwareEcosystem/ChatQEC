from PIL import Image
from chatqec.retrieval.hybrid import RetrievedChunk
from chatqec.retrieval.figure_loader import load_chunk_figures


def _chunk(figs):
    return RetrievedChunk(
        chunk_id="c",
        source_id="pdf:x",
        text="t",
        title="T",
        authority_tier="peer_reviewed",
        score=1.0,
        payload={"figures": figs},
    )


def test_loads_existing_figures_capped(tmp_path):
    for n in ("a.png", "b.png", "c.png", "d.png"):
        Image.new("RGB", (100, 100), (1, 2, 3)).save(tmp_path / n)
    chunk = _chunk(["a.png", "b.png", "c.png", "d.png"])
    imgs = load_chunk_figures([chunk], base_dir=tmp_path, max_images=3)
    assert len(imgs) == 3


def test_skips_missing_files(tmp_path):
    imgs = load_chunk_figures([_chunk(["gone.png"])], base_dir=tmp_path)
    assert imgs == []
