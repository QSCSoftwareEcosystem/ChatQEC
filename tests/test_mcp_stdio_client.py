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


def test_server_subprocess_inherits_parent_env_vars(monkeypatch):
    # Regression guard: mcp.client.stdio.StdioServerParameters defaults to a
    # minimal env (HOME/LOGNAME/PATH/SHELL/TERM/USER only) unless env= is passed
    # explicitly. A prior version of _worker() omitted env=, so a secret set in
    # the parent process (e.g. QISKIT_IBM_TOKEN via .env + load_dotenv()) never
    # reached the spawned chatqec-mcp-tools subprocess at all -- it reported
    # missing credentials even though the parent process had the token. Verify
    # the actual subprocess sees a custom env var, using a real spawn (not a
    # mock of StdioServerParameters) so the fix is checked end to end.
    probe_value = "chatqec-mcp-stdio-env-probe-12345"
    monkeypatch.setenv("QISKIT_IBM_TOKEN", probe_value)

    c = StdioMCPClient(server_command="chatqec-mcp-tools")
    try:
        # ibm_list_backends surfaces the token indirectly: with a syntactically
        # present (if fake) token, the failure mode changes from "no token
        # configured" to a real (fake-token) authentication error -- which can
        # only happen if the subprocess actually received QISKIT_IBM_TOKEN.
        result = c.call_tool("ibm_list_backends", {})
    finally:
        c.close()

    assert result["ok"] is False
    message = result["error_message"].lower()
    # "unable to find account ... saved" is IBM's specific failure when NO
    # token reaches the SDK at all (the symptom this test guards against,
    # observed directly when QISKIT_IBM_TOKEN wasn't forwarded to the
    # subprocess). A forwarded-but-fake token instead fails at IBM's API with a
    # different message (e.g. "unable to retrieve instances" / an HTTP/auth
    # error), never this one.
    assert "find account" not in message
    assert "not installed" not in message
