"""Corpus statistics computation."""

from chatqec.stats import approx_tokens, compute_corpus_stats


def _chunk(text: str, source_id: str, stype: str = "pdf", tier: str = "peer_reviewed", cid=""):
    return {
        "text": text,
        "source_id": source_id,
        "source_type": stype,
        "authority_tier": tier,
        "chunk_id": cid or f"{source_id}#0",
        "title": "T",
    }


def test_approx_tokens_is_ceil_chars_over_4():
    assert approx_tokens("") == 0
    assert approx_tokens("a") == 1
    assert approx_tokens("a" * 4) == 1
    assert approx_tokens("a" * 5) == 2


def test_empty_corpus():
    s = compute_corpus_stats([])
    assert s["n_chunks"] == 0
    assert s["n_sources"] == 0
    assert s["largest_chunk"] is None
    assert s["tokens"]["total"] == 0


def test_stats_counts_sources_tokens_and_largest():
    chunks = [
        _chunk("a" * 40, "pdf:aaa", cid="pdf:aaa#0"),  # 10 tokens
        _chunk("b" * 80, "pdf:aaa", cid="pdf:aaa#1"),  # 20 tokens
        _chunk("c" * 400, "arxiv:1", stype="arxiv", tier="preprint", cid="arxiv:1#0"),  # 100 tok
    ]
    s = compute_corpus_stats(chunks)
    assert s["n_chunks"] == 3
    assert s["n_sources"] == 2
    assert s["tokens"]["total"] == 130
    assert s["tokens"]["max"] == 100
    assert s["largest_chunk"]["chunk_id"] == "arxiv:1#0"
    assert s["by_source_type"] == {"pdf": 2, "arxiv": 1}
    assert s["by_tier"] == {"peer_reviewed": 2, "preprint": 1}
    # per-source aggregation
    assert s["per_source"]["pdf:aaa"]["total"] == 30
    assert s["chunks_per_source"]["max"] == 2
