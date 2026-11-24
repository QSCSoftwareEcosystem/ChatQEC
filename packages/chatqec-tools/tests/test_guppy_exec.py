"""run_isolated: the subprocess runner behind guppy_run (limits, scrubbed env, group kill)."""

import os
import time

import pytest

from chatqec_mcp_tools.tools._guppy_exec import run_isolated

_LIMITS = {"wall_seconds": 20, "cpu_seconds": 20, "memory_mb": 1024}


def test_runs_code_and_captures_stdout():
    r = run_isolated("print('hello')", **_LIMITS)
    assert r.exit_code == 0 and r.stdout.strip() == "hello" and not r.timed_out


def test_environment_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("CHATQEC_TEST_SECRET", "hunter2")
    r = run_isolated("import os\nprint(os.environ.get('CHATQEC_TEST_SECRET'))", **_LIMITS)
    assert r.stdout.strip() == "None"


def test_cwd_is_a_private_directory_that_is_removed_afterwards():
    r = run_isolated("import os\nprint(os.getcwd())", **_LIMITS)
    path = r.stdout.strip()
    assert "chatqec-guppy-" in path and not os.path.exists(path)


def test_timeout_kills_the_whole_process_group():
    code = (
        "import subprocess, time\n"
        "p = subprocess.Popen(['sleep', '300'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    r = run_isolated(code, wall_seconds=2, cpu_seconds=20, memory_mb=1024)
    assert r.timed_out
    child = int(r.stdout.split()[0])
    for _ in range(20):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"grandchild process {child} survived the timeout")


def test_output_is_truncated():
    r = run_isolated("print('x' * 300_000)", output_limit_kb=10, **_LIMITS)
    assert r.truncated and len(r.stdout) < 11 * 1024


def test_memory_limit_is_enforced():
    r = run_isolated(
        "x = bytearray(2 * 2**30)\nprint('allocated')",
        wall_seconds=20,
        cpu_seconds=20,
        memory_mb=512,
    )
    assert "allocated" not in r.stdout and r.exit_code != 0


def test_cpu_limit_stops_a_busy_loop_before_the_wall_clock():
    start = time.monotonic()
    r = run_isolated("while True:\n    pass", wall_seconds=30, cpu_seconds=1, memory_mb=1024)
    assert not r.timed_out and r.exit_code != 0 and time.monotonic() - start < 15


def test_file_size_limit_is_enforced():
    code = "open('big.bin', 'wb').write(b'0' * (5 * 2**20))\nprint('wrote')"
    r = run_isolated(code, fsize_mb=1, **_LIMITS)
    assert "wrote" not in r.stdout


def test_guppy_comptime_eval_is_disabled_in_the_child():
    """Defense in depth behind the source gate: Guppy's comptime(...) runs eval() with
    full builtins at compile time, so the child replaces that evaluator before the
    program runs. A future gate bypass then fails to compile instead of running Python."""
    pytest.importorskip("guppylang_internals.checker.expr_checker")
    code = (
        "import guppylang_internals.checker.expr_checker as ec\n"
        "try:\n"
        "    ec.eval_comptime_expr(None, None)\n"
        "except RuntimeError as e:\n"
        "    print('blocked:', e)\n"
    )
    r = run_isolated(code, **_LIMITS)
    assert r.exit_code == 0, r.stderr
    assert r.stdout.startswith("blocked:") and "comptime" in r.stdout
