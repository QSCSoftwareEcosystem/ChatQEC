"""Eval metric tests."""

from unittest.mock import MagicMock

from chatqec.api import Citation
from chatqec.eval.metrics import (
    citation_accuracy,
    faithfulness_via_judge,
    recall_at_k,
)
from chatqec.retrieval.hybrid import RetrievedChunk


def _r(i: int, source_id: str = "arxiv:test") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_id}#{i}",
        source_id=source_id,
        text="t",
        title="T",
        authority_tier="preprint",
        score=0.5,
        payload={},
    )


def test_recall_at_k_returns_one_when_any_expected_prefix_present():
    retrieved = [_r(0, "arxiv:1808.02892"), _r(1, "blog:xx")]
    assert recall_at_k(retrieved, expected_source_id_prefixes=["arxiv:1808.02892"]) == 1.0


def test_recall_at_k_returns_zero_when_none_present():
    retrieved = [_r(0, "arxiv:other"), _r(1, "blog:xx")]
    assert recall_at_k(retrieved, expected_source_id_prefixes=["arxiv:1808.02892"]) == 0.0


def test_recall_at_k_handles_empty_expected():
    """If no expected sources, recall is trivially 1.0 (nothing to miss)."""
    assert recall_at_k([_r(0)], expected_source_id_prefixes=[]) == 1.0


def test_recall_at_k_handles_prefix_match():
    retrieved = [_r(0, "canonical:surface-code")]
    assert recall_at_k(retrieved, expected_source_id_prefixes=["canonical:surface"]) == 1.0


def test_citation_accuracy_counts_pointing_to_expected_sources():
    cites = [
        Citation(source_id="arxiv:1808.02892", chunk_id="c1", title="t", url="u"),
        Citation(source_id="arxiv:other", chunk_id="c2", title="t", url="u"),
    ]
    score = citation_accuracy(cites, expected_source_id_prefixes=["arxiv:1808.02892"])
    assert score == 0.5


def test_citation_accuracy_zero_for_no_citations():
    assert citation_accuracy([], expected_source_id_prefixes=["x"]) == 0.0


def test_faithfulness_returns_one_when_judge_says_yes():
    client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    client.messages.create.return_value = judge_msg

    score = faithfulness_via_judge(
        question="q",
        answer="a",
        evidence_text="e",
        client=client,
        model="claude-haiku-4-5",
    )
    assert score == 1.0


def test_faithfulness_returns_zero_when_judge_says_no():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="no")]
    client.messages.create.return_value = msg
    assert (
        faithfulness_via_judge(
            question="q",
            answer="a",
            evidence_text="e",
            client=client,
            model="claude-haiku-4-5",
        )
        == 0.0
    )
