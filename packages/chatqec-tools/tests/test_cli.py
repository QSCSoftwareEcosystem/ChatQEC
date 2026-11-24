"""``chatqec-tools`` CLI: list and call, one process per call, JSON on stdout only."""

import json
import subprocess
import sys

from chatqec_mcp_tools.server import _TOOLS, build_server


def _cli(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chatqec_mcp_tools.cli", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_list_matches_registered_tools():
    proc = _cli("list")
    assert proc.returncode == 0, proc.stderr
    listed = json.loads(proc.stdout)
    assert [t["name"] for t in listed] == [t.name for t in _TOOLS]
    assert all(set(t) == {"name", "description", "input_schema"} for t in listed)


def test_call_matches_in_process_invoke():
    args = {"circuit": "X 0\nM 0", "shots": 16}
    proc = _cli("call", "stim_simulate", stdin=json.dumps(args))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == json.loads(
        json.dumps(build_server().invoke_tool("stim_simulate", args), default=str)
    )


def test_stdout_holds_exactly_one_json_line():
    proc = _cli("call", "stim_simulate", stdin=json.dumps({"circuit": "M 0", "shots": 1}))
    assert proc.stdout.count("\n") == 1


def test_bad_arguments_are_a_failed_result_not_a_crash():
    proc = _cli("call", "stim_simulate", stdin=json.dumps({"shots": "many"}))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is False
    assert out["error_message"]


def test_unknown_tool_exits_2():
    proc = _cli("call", "no_such_tool", stdin="{}")
    assert proc.returncode == 2
    assert "unknown tool" in proc.stderr


def test_non_object_arguments_exit_2():
    proc = _cli("call", "stim_simulate", stdin="[1, 2]")
    assert proc.returncode == 2
