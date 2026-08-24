"""Externally supplied chunks compete in the same rerank pass as Qdrant hits."""

from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.hybrid import HybridRetriever, RetrievedChunk


class FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def dense_search(self, *, query_vector, top_k):
        return self._hits

    def sparse_search(self, *, sparse_query, top_k):
        return []


class FakeHit:
    def __init__(self, chunk_id, text, score=1.0):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = {
            "text": text,
            "source_id": chunk_id,
            "title": "corpus",
            "authority_tier": "preprint",
        }


class FakeEmbedder:
    def embed_batch(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeSparse:
    def encode_batch(self, texts):
        return []


class PickByKeyword:
    """Stands in for the cross-encoder: ranks by whether 'wiki' appears."""

    def rerank_scored(self, query, texts, top_k):
        scored = sorted(
            ((i, 10.0 if "wiki" in t else 1.0) for i, t in enumerate(texts)),
            key=lambda p: -p[1],
        )
        return scored[:top_k]


def _retriever(hits):
    return HybridRetriever(
        store=FakeStore(hits),
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparse(),
        reranker=PickByKeyword(),
        canonical_index=CanonicalIndex(pages=[]),
    )


def _wiki_chunk(text="wiki passage about codes"):
    return RetrievedChunk(
        chunk_id="qappswiki:concepts/qec/steane#Description",
        source_id="qappswiki:concepts/qec/steane",
        text=text,
        title="Steane code",
        authority_tier="wiki_unverified",
        score=4.2,
        payload={"url": "https://example.invalid/steane.md"},
    )


def test_extra_chunks_are_reranked_alongside_corpus_chunks():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=2, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert next(c.source_id for c in result.chunks).startswith("qappswiki:")
    assert len(result.chunks) == 2


def test_extra_chunks_survive_when_the_corpus_returns_nothing():
    r = _retriever([])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert [c.source_id for c in result.chunks] == ["qappswiki:concepts/qec/steane"]


def test_extra_chunks_default_to_empty_and_change_nothing():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10)
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.debug["extra_chunks"] == 0


def test_debug_records_how_many_extras_joined():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert result.debug["extra_chunks"] == 1


def test_wiki_only_path_still_reports_rerank_strength():
    r = _retriever([])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert result.debug["top_rerank_score"] == 10.0


# --------------------------------------------------------------------------- #
# the top rerank score travels by return value, not on the instance
# --------------------------------------------------------------------------- #


def test_rerank_pool_returns_the_top_score_instead_of_stashing_it():
    """Streamlit caches one HybridRetriever across every session thread
    (`chatqec_app/utils.py` wraps `get_chatqec()` in `@st.cache_resource`), so an
    instance attribute written by one query and read by another is a real race.
    The score gates the paid web-search fallback, so a crossed read fires or
    suppresses a Tavily call for the wrong question."""
    r = _retriever([])
    chunks, top = r._rerank_pool("codes", [_wiki_chunk()], top_k=5)
    assert [c.source_id for c in chunks] == ["qappswiki:concepts/qec/steane"]
    assert top == 10.0
    assert not hasattr(r, "_last_top_rerank")


def test_an_empty_pool_reranks_to_nothing_with_a_zero_score():
    assert _retriever([])._rerank_pool("codes", [], top_k=5) == ([], 0.0)


def test_a_concurrent_query_cannot_steal_another_querys_rerank_score():
    """Interleave two retrievals on one shared retriever the way two Streamlit
    session threads would. Each result must carry its own score."""
    r = _retriever([FakeHit("c1", "corpus passage")])
    weak = r.retrieve(queries=["codes"], top_k=5, candidate_k=10)
    strong = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    weak_again = r.retrieve(queries=["codes"], top_k=5, candidate_k=10)
    assert strong.debug["top_rerank_score"] == 10.0
    assert weak.debug["top_rerank_score"] == weak_again.debug["top_rerank_score"] == 1.0
