"""CLI test for `chatqec eval generate`."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from chatqec.api import Answer, ToolCall
from chatqec.cli import main


def test_eval_generate_writes_log_and_candidates(tmp_path: Path, monkeypatch):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        json.dumps(
            {
                "id": "route-stim-001",
                "subtype": "code_execution",
                "question": "Simulate H 0\nM 0 for 100 shots.",
                "complexity": "simple",
                "expected_tools": ["stim_simulate"],
                "target_capability": ["stim"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "run.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"

    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="Roughly even split.",
        tool_calls=[ToolCall(name="stim_simulate", input={"shots": 100}, output="{}")],
        confidence="medium",
    )
    build_calls = []

    def _fake_build_chatqec(*args, **kwargs):
        build_calls.append(kwargs)
        return bot

    monkeypatch.setattr("chatqec.cli._build_chatqec", _fake_build_chatqec)

    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [
        MagicMock(
            text=json.dumps(
                {
                    "verdict": "correct",
                    "verdict_reasoning": "Matches tool output.",
                    "gold_answer_summary": "~50/50 split.",
                    "routing_flag": "match",
                    "routing_flag_reasoning": "stim_simulate invoked as expected.",
                }
            )
        )
    ]
    judge_client.messages.create.return_value = judge_msg
    monkeypatch.setattr("chatqec.cli._make_judge_client", lambda: judge_client)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval",
            "generate",
            "--seeds",
            str(seeds),
            "--log",
            str(log_path),
            "--candidates-out",
            str(candidates_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert build_calls == [{"mcp_server": "chatqec-mcp-tools"}]
    assert log_path.exists()
    assert candidates_path.exists()
    candidate = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
    assert candidate["id"] == "route-stim-001"
    assert candidate["gold_answer_summary"] == "~50/50 split."
