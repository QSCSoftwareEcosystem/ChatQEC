"""Local Nomic embedder via sentence-transformers (mocked in tests)."""

from unittest.mock import MagicMock

import numpy as np

from chatqec.embedding.base import EmbedderConfig
from chatqec.embedding.local import LocalEmbedder


def test_local_embedder_uses_sentence_transformers(monkeypatch):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((3, 768), dtype=np.float32)
    mock_st_class = MagicMock(return_value=mock_model)
    monkeypatch.setattr("chatqec.embedding.local.SentenceTransformer", mock_st_class)

    cfg = EmbedderConfig(
        provider="local", model="", batch_size=8, local_model="nomic-ai/nomic-embed-text-v1.5"
    )
    emb = LocalEmbedder(cfg)
    out = emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert len(out[0]) == 768
    mock_st_class.assert_called_once_with(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )


def test_local_embedder_dim_is_768():
    from chatqec.embedding.base import EmbeddingDim

    assert int(EmbeddingDim.NOMIC_15) == 768
