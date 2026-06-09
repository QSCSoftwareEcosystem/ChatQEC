"""Embedder protocol + Gemini wrapper tests (Gemini mocked)."""

from unittest.mock import MagicMock

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim
from chatqec.embedding.gemini import GeminiEmbedder


def test_embedder_protocol_minimum_interface():
    class Stub:
        dim = 8

        def embed_batch(self, texts):
            return [[0.0] * 8 for _ in texts]

    assert isinstance(Stub(), Embedder)


def test_gemini_embedder_calls_api_in_batches(monkeypatch):
    calls: list[list[str]] = []

    def fake_embed_content(model, content, task_type=None, **kw):
        calls.append(content if isinstance(content, list) else [content])
        n = len(calls[-1])
        return {"embedding": [[float(i)] * 768 for i in range(n)]}

    mock_genai = MagicMock()
    mock_genai.embed_content = fake_embed_content
    monkeypatch.setattr("chatqec.embedding.gemini.genai", mock_genai)

    cfg = EmbedderConfig(
        provider="gemini", model="text-embedding-004", batch_size=3, local_model=""
    )
    emb = GeminiEmbedder(cfg)
    out = emb.embed_batch(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert len(out[0]) == 768
    # Batched 3 + 2
    assert [len(c) for c in calls] == [3, 2]


def test_gemini_embedder_dim_is_768():
    cfg = EmbedderConfig(
        provider="gemini", model="text-embedding-004", batch_size=8, local_model=""
    )
    emb = GeminiEmbedder(cfg)
    assert emb.dim == EmbeddingDim.GEMINI_004 == 768
