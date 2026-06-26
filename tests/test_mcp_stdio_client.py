"""Integration tests for the real MCP stdio client against chatqec-mcp-tools.

Skipped when the sibling MCP server is not installed, so the suite stays
portable. Install it with `uv tool install /path/to/chatqec-mcp-tools`.
"""

import shutil

import pytest

from chatqec.conversation.mcp_stdio import StdioMCPClient

pytestmark = pytest.mark.skipif(
    shutil.which("chatqec-mcp-tools") is None,
    reason="chatqec-mcp-tools server not installed",
)


@pytest.fixture
def client():
    c = StdioMCPClient(server_command="chatqec-mcp-tools")
    yield c
    c.close()


def test_list_tools_advertises_stim_simulate(client):
    names = {t.name for t in client.list_tools()}
    assert "stim_simulate" in names
    assert "stim_diagram" in names


def test_call_tool_runs_a_real_stim_circuit(client):
    # X then measure qubit 0 — deterministically samples a 1.
    result = client.call_tool("stim_simulate", {"circuit": "X 0\nM 0", "shots": 16})
    assert result["ok"] is True
    assert result["shots"] == 16
    assert "summary" in result
