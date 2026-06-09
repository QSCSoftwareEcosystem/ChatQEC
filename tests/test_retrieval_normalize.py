"""Score normalization tests — bringing dense and BM25 scores to [0,1]."""

import pytest

from chatqec.retrieval.normalize import min_max_normalize, identity_if_single


def test_min_max_normalize_maps_to_unit_interval():
    out = min_max_normalize([0.5, 1.5, 2.5, 3.5])
    assert out == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_min_max_normalize_empty_returns_empty():
    assert min_max_normalize([]) == []


def test_min_max_normalize_constant_returns_ones():
    # all-equal scores → all 1.0 (avoid divide-by-zero)
    assert min_max_normalize([0.7, 0.7, 0.7]) == [1.0, 1.0, 1.0]


def test_identity_if_single_returns_single_score_as_one():
    assert identity_if_single([0.42]) == [1.0]


def test_identity_if_single_passes_through_for_multiple():
    out = identity_if_single([0.1, 0.5, 0.9])
    assert out == pytest.approx([0.0, 0.5, 1.0])
