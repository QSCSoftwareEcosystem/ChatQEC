"""CLI test for `chatqec eval run`."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from chatqec.api import Answer
from chatqec.cli import main


def test_eval_run_writes_report(tmp_path: Path, monkeypatch):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        '{"id":"d1","subtype":"definition","question":"q",'
        '"expected_topics":[],"expected_source_id_prefixes":["canonical:x"],'
        '"gold_answer_summary":"s"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    bot = MagicMock()
    bot.ask.return_value = Answer(text="A.", citations=[], confidence="low")
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda *_a, **_kw: bot)

    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    judge_client.messages.create.return_value = judge_msg
    monkeypatch.setattr("chatqec.cli._make_judge_client", lambda **_: judge_client)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval",
            "run",
            "--cases",
            str(qa),
            "--output",
            str(out),
            "--judge-model",
            "claude-haiku-4-5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["total"] == 1
    assert "per_subtype" in report


def test_eval_run_passes_mcp_server_through_to_build_chatqec(tmp_path: Path, monkeypatch):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        '{"id":"d1","subtype":"definition","question":"q",'
        '"expected_topics":[],"expected_source_id_prefixes":[],'
        '"gold_answer_summary":"s"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    bot = MagicMock()
    bot.ask.return_value = Answer(text="A.", citations=[], confidence="low")
    build_calls = []

    def _fake_build_chatqec(*args, **kwargs):
        build_calls.append(kwargs)
        return bot

    monkeypatch.setattr("chatqec.cli._build_chatqec", _fake_build_chatqec)

    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    judge_client.messages.create.return_value = judge_msg
    monkeypatch.setattr("chatqec.cli._make_judge_client", lambda **_: judge_client)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval",
            "run",
            "--cases",
            str(qa),
            "--output",
            str(out),
            "--mcp-server",
            "chatqec-mcp-tools",
        ],
    )
    assert result.exit_code == 0, result.output
    assert build_calls == [{"mcp_server": "chatqec-mcp-tools"}]


def test_eval_run_defaults_mcp_server_to_none(tmp_path: Path, monkeypatch):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        '{"id":"d1","subtype":"definition","question":"q",'
        '"expected_topics":[],"expected_source_id_prefixes":[],'
        '"gold_answer_summary":"s"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    bot = MagicMock()
    bot.ask.return_value = Answer(text="A.", citations=[], confidence="low")
    build_calls = []

    def _fake_build_chatqec(*args, **kwargs):
        build_calls.append(kwargs)
        return bot

    monkeypatch.setattr("chatqec.cli._build_chatqec", _fake_build_chatqec)

    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    judge_client.messages.create.return_value = judge_msg
    monkeypatch.setattr("chatqec.cli._make_judge_client", lambda **_: judge_client)

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "run", "--cases", str(qa), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert build_calls == [{"mcp_server": None}]


def test_eval_run_judge_provider_is_passed_to_the_judge_factory(tmp_path: Path, monkeypatch):
    """Development runs must be able to judge with a local model, not spend API credits."""
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        '{"id":"d1","subtype":"definition","question":"q",'
        '"expected_topics":[],"expected_source_id_prefixes":[],'
        '"gold_answer_summary":"s"}\n',
        encoding="utf-8",
    )
    from unittest.mock import MagicMock

    from chatqec.api_types import Answer

    bot = MagicMock()
    bot.ask.return_value = Answer(text="a")
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: bot)
    seen = {}
    judge = MagicMock()
    judge.messages.create.return_value = MagicMock(content=[MagicMock(text="yes")])

    def factory(**kwargs):
        seen.update(kwargs)
        return judge

    monkeypatch.setattr("chatqec.cli._make_judge_client", factory)
    result = CliRunner().invoke(
        main,
        ["eval", "run", "--cases", str(qa), "--output", str(tmp_path / "o.json"),
         "--judge-provider", "local", "--judge-model", "gemma-4"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen == {"provider": "local", "model": "gemma-4"}
