"""Tests for the LaTeX source cleaner used by full-text arXiv ingestion."""

from chatqec.ingestion.latex import (
    clean_latex_body,
    extract_document_body,
    strip_comments,
)


def test_strip_comments_removes_line_comments():
    src = "real text % a comment\nmore text"
    assert strip_comments(src) == "real text \nmore text"


def test_strip_comments_keeps_escaped_percent():
    src = r"50\% of qubits % drop this"
    assert strip_comments(src) == "50\\% of qubits "


def test_extract_document_body_drops_preamble():
    src = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\begin{document}\n"
        "Body content here.\n"
        "\\end{document}\n"
        "trailing junk"
    )
    body = extract_document_body(src)
    assert "documentclass" not in body
    assert "usepackage" not in body
    assert "trailing junk" not in body
    assert "Body content here." in body


def test_extract_document_body_falls_back_to_whole_text():
    # A fragment with no \begin{document} (e.g. an \input'd section file).
    src = "Just a section fragment with $x^2$."
    assert extract_document_body(src) == src


def test_clean_preserves_inline_and_display_math():
    src = "\\begin{document}\nThe stabilizer $S = X_1 X_2$ and $$H\\psi = E\\psi$$\n\\end{document}"
    out = clean_latex_body(src)
    assert "$S = X_1 X_2$" in out
    assert "$$H\\psi = E\\psi$$" in out


def test_clean_preserves_equation_and_code_environments():
    src = (
        "\\begin{document}\n"
        "\\begin{equation} d = 2t + 1 \\end{equation}\n"
        "\\begin{lstlisting}\nstim.Circuit()\n\\end{lstlisting}\n"
        "\\end{document}"
    )
    out = clean_latex_body(src)
    assert "\\begin{equation} d = 2t + 1 \\end{equation}" in out
    assert "stim.Circuit()" in out
    assert "\\begin{lstlisting}" in out


def test_clean_drops_includegraphics_but_keeps_caption():
    src = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\includegraphics[width=0.5\\textwidth]{surface_code.png}\n"
        "\\caption{The surface code lattice.}\n"
        "\\end{figure}\n"
        "\\end{document}"
    )
    out = clean_latex_body(src)
    assert "surface_code.png" not in out
    assert "includegraphics" not in out
    assert "The surface code lattice." in out


def test_clean_unwraps_prose_formatting():
    src = "\\begin{document}\nThis is \\textbf{important} and \\emph{subtle}.\n\\end{document}"
    out = clean_latex_body(src)
    assert "important" in out
    assert "subtle" in out
    assert "textbf" not in out
    assert "emph" not in out


def test_clean_reduces_cite_to_readable_marker():
    src = "\\begin{document}\nShown by Kitaev \\cite{kitaev1997}.\n\\end{document}"
    out = clean_latex_body(src)
    assert "kitaev1997" in out
    assert "\\cite" not in out
