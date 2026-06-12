"""Gemini embedder.

Uses ``gemini-embedding-001`` (GA) — the Matryoshka-truncatable replacement for
the retired ``text-embedding-004``. We request ``output_dimensionality=768`` so
the produced vectors match the project's Qdrant collection (vector_dim=768)
unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

with warnings.catch_warnings():
    # The google.generativeai package emits a FutureWarning on import announcing
    # its deprecation. We pin to it deliberately for v1; silence so it does not
    # trip `filterwarnings = ["error"]` in the test suite.
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim


class GeminiEmbedder(Embedder):
    dim: int = int(EmbeddingDim.GEMINI_004)

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            resp = genai.embed_content(
                model=f"models/{self.cfg.model}",
                content=batch,
                task_type="retrieval_document",
                output_dimensionality=self.dim,
            )
            vectors = resp["embedding"] if isinstance(resp, dict) else resp.embedding
            out.extend(vectors)
        return out
