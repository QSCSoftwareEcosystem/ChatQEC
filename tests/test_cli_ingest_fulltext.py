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
            # This test isolates LaTeX text chunking; opt out of the hybrid figure
            # harvest (extra PDF download + Marker + vision model — network/GPU).
            "--no-describe-figures",
        ],
    )
    assert result.exit_code == 0, result.output

    files = list(chunk_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == len(chunks)

    reg_text = registry_path.read_text()
    assert "arxiv:2408.13687" in reg_text
    assert "pending_review" in reg_text


def test_ingest_arxiv_describe_figures_appends_figure_chunks(tmp_path: Path, monkeypatch):
    """With --describe-figures the LaTeX chunks are kept and figure chunks appended."""
    from chatqec.ingestion.chunk import Chunk

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    registry_path = tmp_path / "registry.jsonl"

    md = ArxivMetadata(
        arxiv_id="2408.13687",
        title="QEC below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        abstract="short abstract",
        pdf_url="",
    )
    text_chunks = paper_to_chunks(
        md, body_text="The surface code. " * 400, target_tokens=512, overlap_tokens=64
    )
    monkeypatch.setattr(
        "chatqec.ingestion.fetchers.arxiv_source.fetch_arxiv_fulltext",
        lambda arxiv_id, **kw: (md, text_chunks),
    )

    # Stub the figure harvest (real one needs network + Marker + GPU); it must be
    # called with start_index == len(text_chunks) and its output appended.
    captured = {}

    def _fake_harvest(arxiv_id, *, source_id, title, describer, start_index, **kw):
        captured["start_index"] = start_index
        return [
            Chunk(
                chunk_id=f"{source_id}#{start_index}",
                source_id=source_id,
                chunk_index=start_index,
                text="Figure (auto-described): a lattice.",
                title=title,
                source_type="arxiv",
                authority_tier="preprint",
                metadata={"figures": ["data/figures/arxiv_2408.13687/f.png"]},
            )
        ]

    monkeypatch.setattr(
        "chatqec.ingestion.fetchers.arxiv_figures.fetch_arxiv_figure_chunks", _fake_harvest
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
            "--describe-figures",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["start_index"] == len(text_chunks)
    lines = (chunk_dir / "arxiv_2408.13687.jsonl").read_text().splitlines()
    assert len(lines) == len(text_chunks) + 1
    assert "auto-described" in lines[-1]
