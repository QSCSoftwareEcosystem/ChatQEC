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
