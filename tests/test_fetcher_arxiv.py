"""arXiv fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.fetchers.arxiv import (
    ArxivMetadata,
    parse_arxiv_atom,
    paper_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_atom_2408.13687.xml"


def test_parse_arxiv_atom_extracts_metadata():
    text = FIXTURE.read_text(encoding="utf-8")
    md = parse_arxiv_atom(text)
    assert isinstance(md, ArxivMetadata)
    assert md.arxiv_id == "2408.13687v1"
    assert md.title.startswith("Quantum error correction")
    assert md.authors[0] == "Rajeev Acharya"
    assert md.year == 2024
    assert "fault-tolerant" in md.abstract.lower()


def test_paper_to_chunks_produces_chunks_with_correct_ids():
    md = ArxivMetadata(
        arxiv_id="2408.13687v1",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        abstract="A long abstract about QEC. " * 100,
        pdf_url="http://arxiv.org/pdf/2408.13687v1",
    )
    chunks = paper_to_chunks(md, body_text=md.abstract, target_tokens=80, overlap_tokens=10)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source_id == "arxiv:2408.13687v1"
    assert chunks[0].chunk_id == "arxiv:2408.13687v1#0"
    assert chunks[0].title.startswith("Quantum error")
    assert chunks[0].source_type == "arxiv"
    assert chunks[0].authority_tier == "preprint"


def test_paper_to_chunks_uses_math_aware_chunker():
    md = ArxivMetadata(
        arxiv_id="abc.1234v1",
        title="t",
        authors=["X"],
        year=2024,
        abstract="",
        pdf_url="",
    )
    body = r"Lead. \begin{equation} S = \langle g_1, g_2 \rangle \end{equation}" + " tail. " * 200
    chunks = paper_to_chunks(md, body_text=body, target_tokens=50, overlap_tokens=5)
    for c in chunks:
        if r"\begin{equation}" in c.text:
            assert r"\end{equation}" in c.text
