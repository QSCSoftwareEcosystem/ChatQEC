"""Cross-encoder reranker tests — model load mocked."""

from unittest.mock import MagicMock

import numpy as np

from chatqec.retrieval.reranker import CrossEncoderReranker


def test_reranker_loads_model_lazily(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.9, 0.1, 0.5])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker(model_name="mixedbread-ai/mxbai-rerank-base-v1")
    mock_ce_class.assert_not_called()  # not loaded yet
    out = rr.rerank("query", ["a", "b", "c"], top_k=2)
    mock_ce_class.assert_called_once_with("mixedbread-ai/mxbai-rerank-base-v1")
    assert out == [0, 2]  # indices of top-2 by score (0.9, 0.5)


def test_reranker_top_k_respects_input_size(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.5, 0.7])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker()
    out = rr.rerank("q", ["x", "y"], top_k=10)
    assert out == [1, 0]


def test_reranker_returns_empty_for_empty_candidates():
    rr = CrossEncoderReranker()
    assert rr.rerank("q", [], top_k=5) == []


def test_rerank_scored_returns_indices_and_scores(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.9, 0.1, 0.5])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker(model_name="mixedbread-ai/mxbai-rerank-base-v1")
    out = rr.rerank_scored("query", ["a", "b", "c"], top_k=2)
    assert out == [(0, 0.9), (2, 0.5)]


def test_rerank_scored_empty_candidates():
    rr = CrossEncoderReranker()
    assert rr.rerank_scored("q", [], top_k=5) == []


def test_rerank_delegates_to_rerank_scored(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.2, 0.8])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker()
    assert rr.rerank("q", ["x", "y"], top_k=2) == [1, 0]


def test_reranker_default_is_non_china_model():
    # Guard the provenance choice: default must not be a BAAI/bge model.
    rr = CrossEncoderReranker()
    assert rr.model_name == "mixedbread-ai/mxbai-rerank-base-v1"
    assert "bge" not in rr.model_name.lower()
    assert "baai" not in rr.model_name.lower()


def _hub_401():
    from huggingface_hub.errors import RepositoryNotFoundError

    response = MagicMock(status_code=401, headers={})
    return RepositoryNotFoundError("401 Client Error", response=response)


def test_reranker_falls_back_to_local_cache_on_hub_auth_error(monkeypatch):
    """A stale HF_TOKEN in the shell makes the hub return 401 even for a public,
    already-cached model. The reranker must then load from the local cache
    instead of failing the whole question (seen in the Streamlit app)."""
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.2, 0.8])
    mock_ce_class = MagicMock(side_effect=[_hub_401(), mock_ce])
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    out = CrossEncoderReranker(model_name="m").rerank("q", ["a", "b"], top_k=1)

    assert out == [1]
    assert mock_ce_class.call_args_list[1].kwargs == {"local_files_only": True}


def test_reranker_reraises_when_model_not_cached_either(monkeypatch):
    import pytest

    mock_ce_class = MagicMock(side_effect=[_hub_401(), OSError("not in cache")])
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    with pytest.raises(Exception, match="401"):
        CrossEncoderReranker(model_name="m").rerank("q", ["a"], top_k=1)
