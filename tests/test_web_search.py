"""Tavily web_search — all HTTP mocked; never hits the network."""

from unittest.mock import MagicMock, patch

from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk
from chatqec.retrieval.web_search import WebResult, build_web_chunks, web_search

_FAKE = {
    "results": [
        {"title": "Surface codes", "url": "https://ex.com/a", "content": "text A", "score": 0.9},
        {"title": "qLDPC", "url": "https://ex.com/b", "content": "text B", "score": 0.7},
    ]
}


def test_web_search_parses_results():
    resp = MagicMock(status_code=200)
    resp.json.return_value = _FAKE
    with patch("requests.post", return_value=resp) as post:
        out = web_search("surface code", k=2, api_key="tvly-x")
    assert [r.url for r in out] == ["https://ex.com/a", "https://ex.com/b"]
    assert out[0] == WebResult(
        title="Surface codes", url="https://ex.com/a", content="text A", score=0.9
    )
    # api_key + query + max_results were sent
    _, kwargs = post.call_args
    assert kwargs["json"]["max_results"] == 2


def test_web_search_empty_key_returns_empty_without_calling():
    with patch("requests.post") as post:
        assert web_search("q", k=3, api_key="") == []
    post.assert_not_called()


def test_web_search_non_200_returns_empty():
    resp = MagicMock(status_code=401)
    resp.json.return_value = {}
    with patch("requests.post", return_value=resp):
        assert web_search("q", k=3, api_key="tvly-x") == []


def test_web_search_request_exception_returns_empty():
    import requests

    with patch("requests.post", side_effect=requests.RequestException("boom")):
        assert web_search("q", k=3, api_key="tvly-x") == []


def test_web_search_malformed_json_returns_empty():
    resp = MagicMock(status_code=200)
    resp.json.side_effect = ValueError("bad json")
    with patch("requests.post", return_value=resp):
        assert web_search("q", k=3, api_key="tvly-x") == []


def test_web_search_skips_result_with_nonnumeric_score():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "results": [
            {"title": "bad", "url": "u1", "content": "c", "score": "N/A"},
            {"title": "ok", "url": "u2", "content": "c", "score": 0.5},
        ]
    }
    with patch("requests.post", return_value=resp):
        out = web_search("q", k=2, api_key="tvly-x")
    assert [r.url for r in out] == ["u2"]  # malformed item skipped, no raise


def _chunk(score):
    return RetrievedChunk(
        chunk_id="c#0",
        source_id="c",
        text="t",
        title="T",
        authority_tier="preprint",
        score=score,
        payload={},
    )


def _stub_searcher(_q, _k):
    return [WebResult(title="W", url="https://ex.com/x", content="web text", score=0.5)]


def _result(chunks, topics, top_rerank_score=0.6):
    # RetrievalResult(chunks, matched_topics, debug={...}) — the gate reads the
    # ABSOLUTE cross-encoder rerank score from debug, not the normalized chunk score.
    return RetrievalResult(
        chunks=chunks, matched_topics=topics, debug={"top_rerank_score": top_rerank_score}
    )


def test_gate_searches_when_retrieval_is_weak():
    out = build_web_chunks(
        "q",
        retrieval=_result([_chunk(0.2)], [], top_rerank_score=0.05),
        web_searcher=_stub_searcher,
        threshold=0.2,
        web_k=5,
    )
    assert len(out) == 1
    assert out[0].authority_tier == "web"
    assert out[0].source_id.startswith("web:")
    assert out[0].payload["url"] == "https://ex.com/x"


def test_gate_searches_when_no_chunks():
    out = build_web_chunks(
        "q",
        retrieval=_result([], [], top_rerank_score=0.0),
        web_searcher=_stub_searcher,
        threshold=0.2,
        web_k=5,
    )
    assert len(out) == 1


def test_gate_skips_when_retrieval_is_strong():
    out = build_web_chunks(
        "q",
        retrieval=_result([_chunk(0.8)], [], top_rerank_score=0.6),
        web_searcher=_stub_searcher,
        threshold=0.2,
        web_k=5,
    )
    assert out == []


def test_gate_skips_when_canonical_topic_matched():
    out = build_web_chunks(
        "q",
        retrieval=_result([_chunk(0.1)], ["surface-code"], top_rerank_score=0.05),
        web_searcher=_stub_searcher,
        threshold=0.2,
        web_k=5,
    )
    assert out == []


def _raising_searcher(_q, _k):
    raise RuntimeError("searcher blew up")


def test_gate_never_raises_when_searcher_raises():
    out = build_web_chunks(
        "q",
        retrieval=_result([], [], top_rerank_score=0.0),
        web_searcher=_raising_searcher,
        threshold=0.2,
        web_k=5,
    )
    assert out == []


def _searcher_with_bad_url(_q, _k):
    return [
        WebResult(title="bad", url=None, content="c", score=0.5),  # type: ignore[arg-type]
        WebResult(title="ok", url="https://ex.com/y", content="c", score=0.5),
    ]


def test_gate_skips_result_with_nonstr_url_without_raising():
    out = build_web_chunks(
        "q",
        retrieval=_result([], [], top_rerank_score=0.0),
        web_searcher=_searcher_with_bad_url,
        threshold=0.2,
        web_k=5,
    )
    assert len(out) == 1
    assert out[0].payload["url"] == "https://ex.com/y"


def _searcher_with_duplicate_urls(_q, _k):
    return [
        WebResult(title="A", url="https://ex.com/dup", content="first", score=0.6),
        WebResult(title="A again", url="https://ex.com/dup", content="second", score=0.4),
    ]


def test_gate_dedupes_results_by_url():
    out = build_web_chunks(
        "q",
        retrieval=_result([], [], top_rerank_score=0.0),
        web_searcher=_searcher_with_duplicate_urls,
        threshold=0.2,
        web_k=5,
    )
    assert len(out) == 1
    assert out[0].text == "first"  # order preserved — first occurrence kept


def test_gate_skips_when_searcher_is_none():
    out = build_web_chunks(
        "q",
        retrieval=_result([], [], top_rerank_score=0.0),
        web_searcher=None,
        threshold=0.2,
        web_k=5,
    )
    assert out == []
