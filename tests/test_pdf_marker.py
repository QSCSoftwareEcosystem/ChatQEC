"""Tests for the isolated-Marker PDF fallback helpers."""

from chatqec.ingestion.fetchers.pdf_marker import (
    build_marker_command,
    find_marker_markdown,
)


def test_build_marker_command_runs_isolated_with_pdf_and_outdir(tmp_path):
    pdf = tmp_path / "paper.pdf"
    out = tmp_path / "out"
    cmd = build_marker_command(pdf, out)

    # Isolated invocation (uvx) of marker's single-file CLI.
    assert "uvx" in cmd
    assert "marker-pdf" in cmd
    assert "marker_single" in cmd
    # Inputs are passed through.
    assert str(pdf) in cmd
    assert str(out) in cmd


def test_find_marker_markdown_picks_nested_md(tmp_path):
    # marker writes <out>/<stem>/<stem>.md
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.md"
    md.write_text("# Title\n\nBody with $X_1$ math.")

    found = find_marker_markdown(tmp_path)
    assert found == md


def test_find_marker_markdown_returns_none_when_absent(tmp_path):
    assert find_marker_markdown(tmp_path) is None
