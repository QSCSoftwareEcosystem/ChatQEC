"""Tavily web_search — all HTTP mocked; never hits the network."""

from unittest.mock import MagicMock, patch

from chatqec.retrieval.web_search import WebResult, web_search

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
