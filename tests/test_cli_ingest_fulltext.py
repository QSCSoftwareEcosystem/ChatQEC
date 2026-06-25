"""CLI test for `chatqec ingest --arxiv <id> --full-text` (LaTeX-source path)."""

from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main
from chatqec.ingestion.fetchers.arxiv import ArxivMetadata, paper_to_chunks


def test_ingest_arxiv_full_text_writes_multiple_chunks(tmp_path: Path, monkeypatch):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    registry_path = tmp_path / "registry.jsonl"

    md = ArxivMetadata(
        arxiv_id="2408.13687",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        abstract="short abstract",
        pdf_url="",
    )
    # A realistic full-text body chunks into many pieces (vs. 1 for the abstract).
    body = "The surface code suppresses logical errors exponentially. " * 400
    chunks = paper_to_chunks(md, body_text=body, target_tokens=512, overlap_tokens=64)
    assert len(chunks) > 1  # sanity: full text is multi-chunk

    monkeypatch.setattr(
        "chatqec.ingestion.fetchers.arxiv_source.fetch_arxiv_fulltext",
        lambda arxiv_id, **kw: (md, chunks),
    )

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(chunk_dir),
            "--registry",
            str(registry_path),
            "--arxiv",
            "2408.13687",
            "--full-text",
        ],
    )
    assert result.exit_code == 0, result.output

    files = list(chunk_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == len(chunks)

    reg_text = registry_path.read_text()
    assert "arxiv:2408.13687" in reg_text
    assert "pending_review" in reg_text
