"""Local fallback embedder using sentence-transformers (Nomic by default)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim


class LocalEmbedder(Embedder):
    dim: int = int(EmbeddingDim.NOMIC_15)

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        # trust_remote_code required for Nomic's custom modeling code
        self.model = SentenceTransformer(cfg.local_model, trust_remote_code=True)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        arr: np.ndarray = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
        )
        result: list[list[float]] = arr.astype(np.float32).tolist()
        return result
