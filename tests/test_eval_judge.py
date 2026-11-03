"""Generation-time judge tests."""

import json

import pytest
from unittest.mock import MagicMock

from chatqec.api import ToolCall
from chatqec.eval.judge import judge_and_trim


def _client_returning(payload: dict) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = msg
    return client


def test_judge_and_trim_parses_well_formed_response():
    client = _client_returning(
        {
            "verdict": "correct",
            "verdict_reasoning": "Matches the stim_simulate output exactly.",
            "gold_answer_summary": "100 shots of H 0; M 0 split roughly 50/50.",
            "routing_flag": "match",
            "routing_flag_reasoning": "stim_simulate was invoked as expected.",
        }
    )
    result = judge_and_trim(
        question="Simulate H 0\nM 0 for 100 shots.",
        raw_answer="The circuit was simulated and produced a roughly even split of 0s and 1s.",
        tool_calls=[ToolCall(name="stim_simulate", input={"shots": 100}, output='{"ok": true}')],
        evidence_text="",
        expected_tools=["stim_simulate"],
        client=client,
        model="claude-haiku-4-5",
    )
    assert result.verdict == "correct"
    assert result.routing_flag == "match"
    assert "50/50" in result.gold_answer_summary


def test_judge_and_trim_raises_on_malformed_json():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="not json at all")]
    client.messages.create.return_value = msg

    with pytest.raises(json.JSONDecodeError):
        judge_and_trim(
            question="q",
            raw_answer="a",
            tool_calls=[],
            evidence_text="",
            expected_tools=[],
            client=client,
            model="claude-haiku-4-5",
        )
