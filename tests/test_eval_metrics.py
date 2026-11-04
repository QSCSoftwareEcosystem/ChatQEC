"""Eval metric tests."""

from unittest.mock import MagicMock

from chatqec.api import Citation, ToolCall
from chatqec.eval.metrics import (
    citation_accuracy,
    faithfulness_via_judge,
    numeric_fact_match,
    recall_at_k,
    tool_call_accuracy,
    unsafe_tool_invoked,
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


def _tc(name: str, output: str = "{}") -> ToolCall:
    return ToolCall(name=name, input={}, output=output)


def test_tool_call_accuracy_perfect_match():
    calls = [_tc("stim_simulate"), _tc("pymatching_decode")]
    result = tool_call_accuracy(calls, expected_tools=["stim_simulate", "pymatching_decode"])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["exact_set_match"] == 1.0


def test_tool_call_accuracy_missing_tool():
    calls = [_tc("stim_simulate")]
    result = tool_call_accuracy(calls, expected_tools=["stim_simulate", "pymatching_decode"])
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["exact_set_match"] == 0.0


def test_tool_call_accuracy_wrong_tool_invoked():
    # known-bad case: a plausible sibling tool fires instead of the right one
    calls = [_tc("tsim_simulate")]
    result = tool_call_accuracy(calls, expected_tools=["stim_simulate"])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["exact_set_match"] == 0.0


def test_tool_call_accuracy_empty_expected_returns_empty_dict():
    assert tool_call_accuracy([_tc("stim_simulate")], expected_tools=[]) == {}


def test_unsafe_tool_invoked_flags_ibm_submit_execution():
    assert unsafe_tool_invoked([_tc("ibm_submit_execution")]) is True
    assert unsafe_tool_invoked([_tc("ibm_preview_execution")]) is False


def test_numeric_fact_match_checks_across_all_tool_outputs():
    calls = [_tc("code_params", output='{"n": 9, "k": 1, "stim_code_distance": 3}')]
    result = numeric_fact_match(calls, expected={"n": 9, "k": 1, "stim_code_distance": 3})
    assert result == {"n": True, "k": True, "stim_code_distance": True}


def test_numeric_fact_match_flags_mismatch_and_missing_key():
    calls = [_tc("code_params", output='{"n": 9, "k": 1}')]
    result = numeric_fact_match(calls, expected={"n": 25, "k": 1, "stim_code_distance": 3})
    assert result == {"n": False, "k": True, "stim_code_distance": False}


def test_numeric_fact_match_ignores_unparseable_output():
    calls = [_tc("stim_simulate", output="not json")]
    result = numeric_fact_match(calls, expected={"n": 9})
    assert result == {"n": False}


def test_numeric_fact_match_empty_expected_returns_empty_dict():
    assert numeric_fact_match([_tc("code_params")], expected={}) == {}


def test_numeric_fact_match_reads_python_repr_tool_output():
    """api.py records tool output as str(dict), a Python repr. The metric only tried
    json.loads, so on every real run it found no facts and returned all False."""
    from chatqec.api_types import ToolCall
    from chatqec.eval.metrics import numeric_fact_match

    call = ToolCall("code_params", {}, repr({"ok": True, "n": 9, "k": 1}))
    assert numeric_fact_match([call], expected={"n": 9, "k": 1}) == {"n": True, "k": True}
