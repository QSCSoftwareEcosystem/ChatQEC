"""CLI tests for `chatqec ask` (wired) and `chatqec chat` (REPL)."""

from unittest.mock import MagicMock

from click.testing import CliRunner

from chatqec.api import Answer
from chatqec.cli import main


def test_ask_uses_real_pipeline(monkeypatch):
    bot = MagicMock()
    bot.ask.return_value = Answer(text="Surface code intro.", confidence="medium")
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: bot)

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What is the surface code?"])
    assert result.exit_code == 0
    assert "Surface code intro." in result.output
    bot.ask.assert_called_once_with("What is the surface code?", images=[])


def test_chat_repl_handles_single_turn(monkeypatch):
    bot = MagicMock()
    bot.ask.return_value = Answer(text="Reply.", confidence="medium")
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: bot)

    runner = CliRunner()
    result = runner.invoke(main, ["chat"], input="hello\n/quit\n")
    assert result.exit_code == 0
    assert "Reply." in result.output


def test_chat_repl_quit_command_exits_cleanly(monkeypatch):
    bot = MagicMock()
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: bot)

    runner = CliRunner()
    result = runner.invoke(main, ["chat"], input="/quit\n")
    assert result.exit_code == 0
    bot.ask.assert_not_called()
