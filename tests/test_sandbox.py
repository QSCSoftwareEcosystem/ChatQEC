"""Sandbox runner tests."""

from chatqec_mcp_tools.sandbox import SandboxConfig, SandboxResult, run_python


def test_sandbox_runs_simple_python_and_captures_stdout():
    cfg = SandboxConfig(wall_seconds=5, memory_mb=256, output_limit_kb=10)
    result = run_python("print('hello')", config=cfg)
    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_sandbox_returns_nonzero_on_python_error():
    cfg = SandboxConfig(wall_seconds=5, memory_mb=256, output_limit_kb=10)
    result = run_python("raise RuntimeError('boom')", config=cfg)
    assert result.exit_code != 0
    assert "RuntimeError" in result.stderr


def test_sandbox_times_out():
    cfg = SandboxConfig(wall_seconds=1, memory_mb=128, output_limit_kb=10)
    result = run_python("import time; time.sleep(5)", config=cfg)
    assert result.timed_out is True
    assert result.exit_code != 0


def test_sandbox_truncates_output_over_limit():
    cfg = SandboxConfig(wall_seconds=5, memory_mb=128, output_limit_kb=1)
    # Produce ~2 KB of output
    result = run_python("print('x' * 2048)", config=cfg)
    assert len(result.stdout) <= 1024 + 200  # truncated near 1 KB
    assert result.truncated is True
