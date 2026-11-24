"""CliToolClient: one ``chatqec-tools`` subprocess per call, JSON in and out."""

import shlex
import sys
import time
from pathlib import Path

import pytest

from chatqec.conversation.cli_tools import CliToolClient, is_tools_cli

FAKE = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).with_name('_fake_tools_cli.py')))}"


@pytest.fixture
def client():
    return CliToolClient(command=FAKE, call_timeout=10, tool_timeouts={"slow": 1})


def test_list_tools_reads_the_cli_listing(client):
    tools = client.list_tools()
    assert [t.name for t in tools] == ["echo", "slow"]
    assert tools[0].input_schema == {"type": "object"}


def test_call_passes_args_on_stdin_and_parses_stdout(client, monkeypatch):
    monkeypatch.setenv("FAKE_TOOLS_ENV", "inherited")
    out = client.call_tool("echo", {"distance": 3, "shots": 10})
    assert out == {"ok": True, "args": {"distance": 3, "shots": 10}, "env": "inherited"}


def test_timeout_returns_an_error_result_and_kills_the_call(client):
    start = time.monotonic()
    out = client.call_tool("slow", {})
    assert time.monotonic() - start < 10
    assert out["ok"] is False
    assert "exceeded 1s" in out["error_message"]


def test_non_json_output_is_a_failed_result(client):
    out = client.call_tool("garbage", {})
    assert out["ok"] is False
    assert "non-JSON" in out["error_message"]


def test_nonzero_exit_reports_stderr(client):
    out = client.call_tool("crash", {})
    assert out["ok"] is False
    assert "code 3" in out["error_message"]
    assert "boom" in out["error_message"]


def test_missing_command_raises_at_construction():
    with pytest.raises(FileNotFoundError):
        CliToolClient(command="definitely-not-a-real-chatqec-tools-binary")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("chatqec-tools", True),
        ("/home/u/.local/bin/chatqec-tools", True),
        ("chatqec-mcp-tools", False),
        ("qappswiki serve --root ~/QAppsWiki", False),
        ("", False),
    ],
)
def test_is_tools_cli(command, expected):
    assert is_tools_cli(command) is expected
