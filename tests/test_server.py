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
    out = server.invoke_tool(
        "threshold_sweep",
        {
            "code_family": "repetition_code",
            "distances": [3],
            "p_values": [0.1],
            "shots": 50,
        },
    )
    assert out["ok"] is True
    assert len(out["rows"]) == 1


def test_ibm_list_backends_is_registered():
    server = build_server()
    assert "ibm_list_backends" in {t.name for t in server.list_tools()}


def test_ibm_backend_calibration_is_registered():
    server = build_server()
    assert "ibm_backend_calibration" in {t.name for t in server.list_tools()}


def test_ibm_map_circuit_is_registered():
    server = build_server()
    assert "ibm_map_circuit" in {t.name for t in server.list_tools()}


def test_ibm_preview_execution_is_registered():
    server = build_server()
    assert "ibm_preview_execution" in {t.name for t in server.list_tools()}


def test_ibm_submit_execution_is_registered():
    server = build_server()
    assert "ibm_submit_execution" in {t.name for t in server.list_tools()}


def test_protocol_stdout_is_isolated_from_native_prints():
    """A native library (CUDA-Q printed 'CUDA ERROR csr.h:61: out of memory')
    writing to fd 1 must not corrupt the MCP JSON-RPC stream: after isolation,
    fd 1 goes to stderr and the protocol writes to a private copy of stdout."""
    import subprocess
    import sys

    code = (
        "import os\n"
        "from chatqec_mcp_tools.server import _isolate_protocol_stdout\n"
        "proto = _isolate_protocol_stdout()\n"
        "os.write(1, b'native junk\\n')\n"
        "print('python print')\n"
        'proto.write(\'{"jsonrpc": "2.0"}\\n\'); proto.flush()\n'
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout == '{"jsonrpc": "2.0"}\n'
    assert "native junk" in out.stderr
    assert "python print" in out.stderr
