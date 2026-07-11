"""PDF fetcher tests — operates on pre-extracted text."""

from chatqec.ingestion.fetchers.pdf import (
    PDFMetadata,
    text_to_chunks,
)


def test_text_to_chunks_uses_pdf_source_id():
    md = PDFMetadata(
        content_sha256="ab12cd34" + "0" * 56,
        title="A QEC paper",
        author="Alice",
        year=2024,
        source_path="/tmp/paper.pdf",
    )
    chunks = text_to_chunks(md, body_text="Body. " * 200, target_tokens=80, overlap_tokens=8)
    assert chunks[0].source_id == "pdf:ab12cd34"
    assert chunks[0].source_type == "pdf"
    assert chunks[0].title == "A QEC paper"


def test_text_to_chunks_protects_math():
    md = PDFMetadata(
        content_sha256="a" * 64,
        title="t",
        author="a",
        year=2024,
        source_path="/x.pdf",
    )
    body = r"Lead. \begin{align} S &= \langle g_1, g_2 \rangle \end{align}" + " tail. " * 200
    chunks = text_to_chunks(md, body_text=body, target_tokens=50, overlap_tokens=5)
    for c in chunks:
        if r"\begin{align}" in c.text:
            assert r"\end{align}" in c.text


def test_text_to_chunks_records_figure_paths_in_metadata():
    md = PDFMetadata(content_sha256="a" * 64, title="T", author="A", year=2020, source_path="x.pdf")
    body = "Prose ![](data/figures/pdf_aaaaaaaa/f1.png) more prose."
    chunks = text_to_chunks(md, body_text=body)
    figs = [f for c in chunks for f in c.metadata.get("figures", [])]
    assert "data/figures/pdf_aaaaaaaa/f1.png" in figs
