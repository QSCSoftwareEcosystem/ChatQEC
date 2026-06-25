"""Tests for main-tex discovery and \\input/\\include resolution."""

from chatqec.ingestion.latex import (
    assemble_source,
    find_main_tex,
    resolve_inputs,
)


def test_find_main_tex_picks_file_with_document_env():
    files = {
        "section1.tex": "Some section text with $x$.",
        "main.tex": "\\documentclass{article}\n\\begin{document}\nHi\n\\end{document}",
    }
    assert find_main_tex(files) == "main.tex"


def test_find_main_tex_single_file():
    files = {"paper.tex": "no document env here"}
    assert find_main_tex(files) == "paper.tex"


def test_resolve_inputs_inlines_without_extension():
    files = {
        "main.tex": "Intro.\n\\input{sec1}\nOutro.",
        "sec1.tex": "Section one body.",
    }
    out = resolve_inputs(files["main.tex"], files)
    assert "Section one body." in out
    assert "\\input" not in out


def test_resolve_inputs_inlines_with_extension_and_include():
    files = {
        "main.tex": "\\include{chapters/c1.tex}",
        "chapters/c1.tex": "Chapter one.",
    }
    out = resolve_inputs(files["main.tex"], files)
    assert "Chapter one." in out
    assert "\\include" not in out


def test_resolve_inputs_drops_missing_target():
    files = {"main.tex": "Before.\n\\input{nonexistent}\nAfter."}
    out = resolve_inputs(files["main.tex"], files)
    assert "Before." in out
    assert "After." in out
    assert "\\input" not in out


def test_assemble_source_combines_then_is_cleanable():
    files = {
        "main.tex": (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "Main intro. \\input{sec1}\n"
            "\\end{document}"
        ),
        "sec1.tex": "Stabilizer $S = X_1 X_2$.",
    }
    raw = assemble_source(files)
    assert "Main intro." in raw
    assert "$S = X_1 X_2$" in raw
