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


def test_build_server_registers_new_qec_tools():
    server = build_server()
    tool_names = {t.name for t in server.list_tools()}
    assert {"pymatching_decode", "threshold_sweep"} <= tool_names


def test_threshold_sweep_handler_round_trips():
    server = build_server()
    out = server.invoke_tool("threshold_sweep", {
        "code_family": "repetition_code",
        "distances": [3],
        "p_values": [0.1],
        "shots": 50,
    })
    assert out["ok"] is True
    assert len(out["rows"]) == 1


def test_ibm_list_backends_is_registered():
    server = build_server()
    assert "ibm_list_backends" in {t.name for t in server.list_tools()}


def test_ibm_backend_calibration_is_registered():
    server = build_server()
    assert "ibm_backend_calibration" in {t.name for t in server.list_tools()}
