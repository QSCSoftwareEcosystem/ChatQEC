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


def test_reranker_default_is_non_china_model():
    # Guard the provenance choice: default must not be a BAAI/bge model.
    rr = CrossEncoderReranker()
    assert rr.model_name == "mixedbread-ai/mxbai-rerank-base-v1"
    assert "bge" not in rr.model_name.lower()
    assert "baai" not in rr.model_name.lower()
