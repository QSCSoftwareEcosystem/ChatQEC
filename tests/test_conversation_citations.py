"""Citation extraction: parse [^N] markers and map back to retrieved chunks."""

from chatqec.api import Citation
from chatqec.conversation.citations import extract_citations
from chatqec.retrieval.hybrid import RetrievedChunk


def _chunk(i: int, source_id: str = "arxiv:test") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_id}#{i}",
        source_id=source_id,
        text="t",
        title=f"Title {i}",
        authority_tier="preprint",
        score=0.5,
        payload={"url": f"https://arxiv.org/abs/{source_id}#{i}"},
    )


def test_extract_resolves_footnote_markers_to_chunks():
    answer = "The surface code has high threshold[^1]. The toric code is its torus variant[^2]."
    context_chunks = [_chunk(0), _chunk(1)]
    citations = extract_citations(answer, context_chunks)
    assert [c.source_id for c in citations] == ["arxiv:test", "arxiv:test"]
    assert all(isinstance(c, Citation) for c in citations)


def test_extract_deduplicates_repeated_footnote_numbers():
    answer = "Claim A [^1]. Claim B [^1]. Claim C [^2]."
    citations = extract_citations(answer, [_chunk(0), _chunk(1)])
    assert len(citations) == 2


def test_extract_ignores_out_of_range_footnotes():
    answer = "Claim with bogus citation [^99]."
    citations = extract_citations(answer, [_chunk(0)])
    assert citations == []


def test_extract_empty_for_answer_with_no_markers():
    citations = extract_citations("Plain answer no marker.", [_chunk(0)])
    assert citations == []


def _chunk_with_payload(i: int, source_id: str, payload: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_id}#{i}",
        source_id=source_id,
        text="t",
        title=f"Title {i}",
        authority_tier="preprint",
        score=0.5,
        payload=payload or {},
    )


def test_extract_reads_url_from_meta_url_namespaced_key():
    """Qdrant store namespaces chunk metadata as meta_<key>."""
    chunks = [_chunk_with_payload(0, "url:abc", {"meta_url": "https://example.com/x"})]
    cites = extract_citations("see [^1]", chunks)
    assert cites[0].url == "https://example.com/x"


def test_extract_prefers_plain_url_over_meta_url():
    chunks = [
        _chunk_with_payload(
            0, "url:abc", {"url": "https://primary", "meta_url": "https://secondary"}
        )
    ]
    cites = extract_citations("see [^1]", chunks)
    assert cites[0].url == "https://primary"


def test_extract_derives_arxiv_url_from_source_id_when_payload_missing():
    chunks = [_chunk_with_payload(0, "arxiv:2408.13687", {})]
    cites = extract_citations("see [^1]", chunks)
    assert cites[0].url == "https://arxiv.org/abs/2408.13687"


def test_extract_falls_back_to_pdf_url_when_no_abs_url():
    chunks = [_chunk_with_payload(0, "url:abc", {"meta_pdf_url": "https://example.com/paper.pdf"})]
    cites = extract_citations("see [^1]", chunks)
    assert cites[0].url == "https://example.com/paper.pdf"


def test_extract_returns_empty_url_for_unknown_source_type():
    chunks = [_chunk_with_payload(0, "github:org/repo", {})]
    cites = extract_citations("see [^1]", chunks)
    assert cites[0].url == ""


def test_strip_trailing_sources_removes_block_keeps_inline_markers():
    from chatqec.conversation.citations import strip_trailing_sources

    text = (
        "The surface code detects errors [^1]. It is a CSS code [^2].\n\n"
        "Sources:\n[^1]: [preprint - A - arxiv:1]\n[^2]: [blog - B - url:2]"
    )
    out = strip_trailing_sources(text)
    assert out == "The surface code detects errors [^1]. It is a CSS code [^2]."
    assert "Sources:" not in out
    assert "[^1]" in out  # inline markers preserved


def test_strip_trailing_sources_noop_without_block():
    from chatqec.conversation.citations import strip_trailing_sources

    text = "A plain answer with a marker [^1] and no trailing list."
    assert strip_trailing_sources(text) == text
