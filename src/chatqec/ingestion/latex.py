r"""LaTeX source cleaner for full-text arXiv ingestion.

arXiv ships the original LaTeX source for most papers. That is the highest-
fidelity input for QEC content: equations stay as ``$...$`` / ``\begin{equation}``
and code stays in ``lstlisting``/``verbatim`` environments — exactly the spans
the math-aware chunker already protects.

This module turns raw ``.tex`` into clean, chunk-ready text:

- strip ``%`` comments (but keep escaped ``\%``),
- drop the preamble (keep only ``\begin{document}..\end{document}``),
- drop ``\includegraphics`` images but keep ``\caption{...}`` text,
- unwrap prose-formatting commands (``\textbf{x}`` -> ``x``),
- reduce ``\cite{key}`` to a readable ``[key]`` marker,

while leaving math and code environments byte-for-byte intact.
"""

from __future__ import annotations

import re

# A line comment is an unescaped `%` to end of line. Keep the newline.
_COMMENT_RE = re.compile(r"(?<!\\)%.*")

_BODY_RE = re.compile(r"\\begin\{document\}(.*?)\\end\{document\}", re.DOTALL)

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}")

# Prose-formatting commands whose single brace argument is kept as plain text.
_UNWRAP_CMDS = (
    "textbf",
    "textit",
    "emph",
    "texttt",
    "textrm",
    "textsc",
    "mbox",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
)
_UNWRAP_RE = re.compile(
    r"\\(?:" + "|".join(_UNWRAP_CMDS) + r")\*?\{([^{}]*)\}",
)

# \cite{a,b} / \citep{a} / \citet{a} -> [a,b]
_CITE_RE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])?\{([^}]*)\}")

_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")

# Bound \input/\include recursion to avoid cycles in malformed source.
_MAX_INPUT_DEPTH = 10


def strip_comments(tex: str) -> str:
    """Remove ``%`` line comments, preserving escaped ``\\%`` and the newline."""
    return _COMMENT_RE.sub("", tex)


def extract_document_body(tex: str) -> str:
    """Return the text inside ``\\begin{document}..\\end{document}``.

    Falls back to the whole string when there is no document environment
    (e.g. an ``\\input``'d section fragment).
    """
    m = _BODY_RE.search(tex)
    return m.group(1) if m else tex


def find_main_tex(files: dict[str, str]) -> str:
    """Pick the main ``.tex`` file: the one with ``\\begin{document}``.

    Falls back to ``\\documentclass``, then to the single/first ``.tex`` file.
    """
    tex_files = {n: c for n, c in files.items() if n.endswith(".tex")}
    if not tex_files:
        tex_files = dict(files)
    for name, content in tex_files.items():
        if "\\begin{document}" in content:
            return name
    for name, content in tex_files.items():
        if "\\documentclass" in content:
            return name
    return next(iter(tex_files))


def _lookup(target: str, files: dict[str, str]) -> str | None:
    """Resolve an \\input/\\include target to file content, trying .tex extension."""
    for candidate in (target, f"{target}.tex"):
        if candidate in files:
            return files[candidate]
    return None


def resolve_inputs(text: str, files: dict[str, str], _depth: int = 0) -> str:
    """Recursively inline ``\\input``/``\\include`` targets from ``files``.

    Missing targets are dropped. Recursion is bounded to avoid cycles.
    """
    if _depth > _MAX_INPUT_DEPTH:
        return _INPUT_RE.sub("", text)

    def _repl(m: re.Match[str]) -> str:
        content = _lookup(m.group(1).strip(), files)
        if content is None:
            return ""
        return resolve_inputs(content, files, _depth + 1)

    return _INPUT_RE.sub(_repl, text)


def assemble_source(files: dict[str, str]) -> str:
    """Find the main tex file and inline its inputs into one raw LaTeX string."""
    main = find_main_tex(files)
    return resolve_inputs(files[main], files)


def clean_latex_body(tex: str) -> str:
    """Full cleaning pipeline: comments -> body -> drop graphics -> unwrap prose."""
    text = strip_comments(tex)
    text = extract_document_body(text)
    text = _INCLUDEGRAPHICS_RE.sub("", text)
    text = _CITE_RE.sub(r"[\1]", text)
    # Unwrap prose commands repeatedly to handle (shallow) nesting.
    prev = None
    while prev != text:
        prev = text
        text = _UNWRAP_RE.sub(r"\1", text)
    return text


def source_files_to_text(files: dict[str, str]) -> str:
    """Assemble a multi-file LaTeX source into one clean, chunk-ready string."""
    return clean_latex_body(assemble_source(files))
