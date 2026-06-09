"""BM25 sparse encoder tests (via fastembed)."""

from chatqec.embedding.sparse import BM25Encoder, SparseVector


def test_bm25_returns_index_value_pairs():
    enc = BM25Encoder()
    vecs = enc.encode_batch(["surface code", "color code", "color code variant"])
    assert len(vecs) == 3
    for v in vecs:
        assert isinstance(v, SparseVector)
        assert len(v.indices) == len(v.values)
        assert all(val > 0 for val in v.values)


def test_bm25_term_frequency_saturates():
    enc = BM25Encoder()
    # fastembed's Qdrant/bm25 model is stateless (no cross-corpus IDF), so the
    # meaningful, deterministic property to assert is BM25 term-frequency
    # saturation: repeating a term raises its weight but sub-linearly.
    once, repeated = enc.encode_batch(["code", "code code code code code"])
    w_once = once.values[0]
    w_repeated = repeated.values[0]
    assert w_repeated > w_once  # more occurrences → higher weight
    assert w_repeated < 5 * w_once  # ... but BM25 saturates (sub-linear)


def test_bm25_empty_string_returns_empty_vector():
    enc = BM25Encoder()
    [v] = enc.encode_batch([""])
    assert v.indices == [] or len(v.indices) == 0
