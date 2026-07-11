"""CLI shell smoke tests using Click's CliRunner."""

from unittest.mock import MagicMock

from click.testing import CliRunner

from chatqec.api import Answer
from chatqec.cli import main


def test_cli_ask_prints_figures_section(monkeypatch):
    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="Surface code intro.",
        confidence="medium",
        figure_paths=["data/figures/pdf_x/f1.png"],
    )
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: bot)

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What is the surface code?"])
    assert result.exit_code == 0
    assert "Figures:" in result.output
    assert "data/figures/pdf_x/f1.png" in result.output


def test_cli_help_includes_program_name():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chatqec" in result.output.lower()


def test_cli_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_ask_subcommand_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "--help"])
    assert result.exit_code == 0
    assert "question" in result.output.lower()


def test_cli_chat_subcommand_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "interactive" in result.output.lower() or "repl" in result.output.lower()
