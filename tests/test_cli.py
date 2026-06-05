"""CLI shell smoke tests using Click's CliRunner."""

from click.testing import CliRunner

from chatqec.cli import main


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


def test_cli_ask_with_stub_returns_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What is the surface code?"])
    assert result.exit_code == 0
    assert "stub" in result.output.lower() or "not yet implemented" in result.output.lower()
