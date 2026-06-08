"""Math-aware chunker — never splits inside LaTeX math environments.

The single biggest accuracy lever for QEC content. Tests use golden strings
that contain $...$, $$...$$, \\begin{equation}...\\end{equation}, and
\\begin{align}...\\end{align}.
"""

from chatqec.ingestion.chunkers.math_aware import chunk_text


def test_chunker_yields_chunks_within_target_size():
    text = "qec " * 1000  # ~4000 chars
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=20)
    assert len(chunks) > 1
    # crude proxy for token count (~4 chars/token)
    assert all(len(c) <= 200 * 6 for c in chunks)


def test_chunker_never_splits_inline_math():
    text = (
        "Definition. The stabilizer of a surface code satisfies $S_i S_j = S_j S_i$ "
        + "for all i, j, as proved in earlier work. "
    ) * 200  # long enough to force multiple chunks
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    for c in chunks:
        # No chunk may end in the middle of an inline $...$ math span
        dollar_count = c.count("$")
        # Even count means math spans are paired (closed)
        assert dollar_count % 2 == 0, f"chunk has unbalanced $ count {dollar_count}:\n{c!r}"


def test_chunker_keeps_equation_environment_intact():
    text = (
        "The stabilizer code definition.\n\n"
        + r"\begin{equation}"
        + "\n"
        + r"S = \langle g_1, \ldots, g_{n-k} \rangle \subset P_n"
        + "\n"
        + r"\end{equation}"
        + "\n\n"
        + "and the code space is "
        + ("Hello world. " * 200)
    )
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    # Find the chunk containing \begin{equation}; it must also contain \end{equation}
    found = False
    for c in chunks:
        if r"\begin{equation}" in c:
            assert r"\end{equation}" in c, "equation env split across chunks"
            found = True
    assert found, "test fixture not large enough"


def test_chunker_keeps_align_environment_intact():
    text = (
        "Lead-in. "
        + r"\begin{align}"
        + "\n"
        + "a &= b \\\\\n"
        + "c &= d"
        + "\n"
        + r"\end{align}"
        + "\n"
        + "tail-in. " * 400
    )
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    for c in chunks:
        if r"\begin{align}" in c:
            assert r"\end{align}" in c


def test_chunker_keeps_display_math_intact():
    text = (
        "Lead-in. " * 50
        + "$$\\sum_i a_i = 1$$"
        + " mid. " * 50
        + "$$\\prod_j b_j = 0$$"
        + " tail. " * 50
    )
    chunks = chunk_text(text, target_tokens=40, overlap_tokens=5)
    for c in chunks:
        assert c.count("$$") % 2 == 0


def test_chunker_returns_single_chunk_for_short_text():
    text = "Short body."
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=20)
    assert chunks == [text]


def test_chunker_handles_empty_text():
    assert chunk_text("", target_tokens=200, overlap_tokens=20) == []


def test_chunker_overlap_is_within_target():
    text = "A sentence about QEC. " * 500
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=20)
    if len(chunks) >= 2:
        # Overlap implies consecutive chunks share some prefix/suffix words
        assert any(w in chunks[0] for w in chunks[1].split()[:10])
