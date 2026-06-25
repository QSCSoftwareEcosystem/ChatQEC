"""Tests for arXiv LaTeX-source extraction and full-text assembly."""

import gzip
import io
import tarfile

from chatqec.ingestion.fetchers.arxiv_source import extract_tex_from_tar
from chatqec.ingestion.latex import source_files_to_text


def _make_tar_gz(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def test_extract_tex_from_tar_returns_only_tex_decoded():
    blob = _make_tar_gz(
        {
            "main.tex": b"\\begin{document}\nHello $x$\n\\end{document}",
            "sec1.tex": b"Section body.",
            "figure.png": b"\x89PNG\x00binarygarbage",
        }
    )
    files = extract_tex_from_tar(blob)
    assert set(files) == {"main.tex", "sec1.tex"}
    assert "Hello $x$" in files["main.tex"]


def test_extract_tex_handles_gzipped_single_tex():
    # Some arXiv sources are a single gzipped .tex (no tar wrapper).
    inner = b"\\documentclass{article}\\begin{document}One file $y^2$.\\end{document}"
    blob = gzip.compress(inner)
    files = extract_tex_from_tar(blob)
    assert any("One file $y^2$." in c for c in files.values())


def test_extract_tex_returns_empty_for_pdf_blob():
    # arXiv serves a PDF when no LaTeX source exists — must NOT decode as text.
    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj binary garbage \x00\x01\x02"
    assert extract_tex_from_tar(pdf) == {}


def test_extract_tex_returns_empty_for_unknown_binary():
    blob = bytes(range(256)) * 4  # not tar, not gzip, not text
    assert extract_tex_from_tar(blob) == {}


def test_source_files_to_text_end_to_end():
    files = {
        "main.tex": (
            "\\documentclass{article}\n"
            "% a comment\n"
            "\\begin{document}\n"
            "Intro \\textbf{bold}. \\input{sec1}\n"
            "\\includegraphics{fig.png}\n"
            "\\caption{Cap text.}\n"
            "\\end{document}"
        ),
        "sec1.tex": "Stabilizer $S = X_1 X_2$ \\cite{kitaev}.",
    }
    text = source_files_to_text(files)
    assert "Intro bold." in text
    assert "$S = X_1 X_2$" in text
    assert "[kitaev]" in text
    assert "Cap text." in text
    assert "fig.png" not in text
    assert "a comment" not in text
    assert "documentclass" not in text
