"""MCP server smoke test - verify tool registration without spawning the stdio loop."""

from chatqec_mcp_tools.server import build_server


def test_build_server_registers_both_tools():
    server = build_server()
    tool_names = {t.name for t in server.list_tools()}
    assert "stim_simulate" in tool_names
    assert "stim_diagram" in tool_names


def test_tool_handlers_are_invokable():
    server = build_server()
    out = server.invoke_tool("stim_simulate", {"circuit": "M 0\n", "shots": 1})
    assert out["ok"] is True
