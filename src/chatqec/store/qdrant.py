"""Thin Qdrant wrapper specialised for ChatQEC's chunk schema."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk


@dataclass(frozen=True)
class StoreConfig:
    qdrant_url: str  # http url or ":memory:"
    collection: str
    vector_dim: int
    on_disk_payload: bool


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source_id: str
    score: float
    payload: dict[str, object]


_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


class ChatQecStore:
    """ChatQEC's Qdrant-backed chunk store with hybrid (dense+sparse) vectors."""

    def __init__(self, cfg: StoreConfig) -> None:
        self.cfg = cfg
        if cfg.qdrant_url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=cfg.qdrant_url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.collection_exists():
            return
        self._client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={
                _DENSE_NAME: models.VectorParams(
                    size=self.cfg.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=self.cfg.on_disk_payload,
                ),
            },
            sparse_vectors_config={
                _SPARSE_NAME: models.SparseVectorParams(),
            },
        )
        with warnings.catch_warnings():
            # The local/in-memory Qdrant client warns that payload indexes have
            # no effect; against a real server they are honoured. Silence the
            # local-mode noise so it doesn't trip `filterwarnings = ["error"]`.
            warnings.simplefilter("ignore", UserWarning)
            for field in ("source_id", "source_type", "authority_tier", "status", "topic_slugs"):
                self._client.create_payload_index(
                    collection_name=self.cfg.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def collection_exists(self) -> bool:
        return self.cfg.collection in {c.name for c in self._client.get_collections().collections}

    def count(self) -> int:
        return int(self._client.count(self.cfg.collection, exact=True).count)

    @staticmethod
    def _point_uuid(chunk_id: str) -> str:
        # Qdrant point IDs must be int or UUID; we hash chunk_id → UUID5.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        *,
        chunks: Iterable[Chunk],
        dense_vectors: Iterable[list[float]],
        sparse_vectors: Iterable[SparseVector],
    ) -> None:
        points = []
        for c, dv, sv in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            payload = {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "title": c.title,
                "source_type": c.source_type,
                "authority_tier": c.authority_tier,
                "anchor": c.anchor,
                "status": "active",
                **{f"meta_{k}": v for k, v in c.metadata.items()},
            }
            points.append(
                models.PointStruct(
                    id=self._point_uuid(c.chunk_id),
                    vector={
                        _DENSE_NAME: dv,
                        _SPARSE_NAME: models.SparseVector(
                            indices=sv.indices,
                            values=sv.values,
                        ),
                    },
                    payload=payload,
                )
            )
        if points:
            self._client.upsert(collection_name=self.cfg.collection, points=points)

    def dense_search(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        status_filter: str | None = "active",
    ) -> list[SearchHit]:
        flt = None
        if status_filter:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value=status_filter)
                    ),
                ]
            )
        result = self._client.query_points(
            collection_name=self.cfg.collection,
            query=query_vector,
            using=_DENSE_NAME,
            limit=top_k,
            query_filter=flt,
        )
        hits: list[SearchHit] = []
        for p in result.points:
            pl: dict[str, object] = dict(p.payload or {})
            hits.append(
                SearchHit(
                    chunk_id=str(pl.get("chunk_id", "")),
                    source_id=str(pl.get("source_id", "")),
                    score=p.score,
                    payload=pl,
                )
            )
        return hits

    def sparse_search(
        self,
        *,
        sparse_query: SparseVector,
        top_k: int,
        status_filter: str | None = "active",
    ) -> list[SearchHit]:
        flt = None
        if status_filter:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value=status_filter)
                    ),
                ]
            )
        result = self._client.query_points(
            collection_name=self.cfg.collection,
            query=models.SparseVector(
                indices=sparse_query.indices,
                values=sparse_query.values,
            ),
            using=_SPARSE_NAME,
            limit=top_k,
            query_filter=flt,
        )
        hits: list[SearchHit] = []
        for p in result.points:
            pl: dict[str, object] = dict(p.payload or {})
            hits.append(
                SearchHit(
                    chunk_id=str(pl.get("chunk_id", "")),
                    source_id=str(pl.get("source_id", "")),
                    score=p.score,
                    payload=pl,
                )
            )
        return hits

    def set_status_for_source(self, source_id: str, new_status: str) -> None:
        self._client.set_payload(
            collection_name=self.cfg.collection,
            payload={"status": new_status},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    ),
                ]
            ),
        )

    def delete_by_source_id(self, source_id: str) -> None:
        self._client.delete(
            collection_name=self.cfg.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        ),
                    ]
                )
            ),
        )
