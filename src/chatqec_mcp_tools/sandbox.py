"""Subprocess-based sandbox runner.

v1 uses subprocess + rlimits for portability (no Docker requirement).
v2 will swap in a DockerSandboxRunner with the same interface (see roadmap).
"""

from __future__ import annotations

import contextlib
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxConfig:
    wall_seconds: int = 30
    memory_mb: int = 1024
    output_limit_kb: int = 100


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


def _preexec_set_rlimits(memory_mb: int) -> Callable[[], None]:
    def _set() -> None:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Disable forking (best-effort)
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

    return _set


def _truncate(text: str, limit_kb: int) -> tuple[str, bool]:
    limit = limit_kb * 1024
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n... [truncated]", True


def run_python(code: str, *, config: SandboxConfig) -> SandboxResult:
    """Run user-supplied Python in a sandboxed subprocess with rlimits."""
    with tempfile.TemporaryDirectory(prefix="chatqec-sandbox-") as workdir:
        script = Path(workdir) / "run.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=config.wall_seconds,
                preexec_fn=_preexec_set_rlimits(config.memory_mb),
                env={"PYTHONUNBUFFERED": "1", "HOME": workdir, "PATH": ""},
                check=False,
            )
            stdout, t1 = _truncate(proc.stdout, config.output_limit_kb)
            stderr, t2 = _truncate(proc.stderr, config.output_limit_kb)
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                truncated=(t1 or t2),
            )
        except subprocess.TimeoutExpired as e:
            raw_stdout = e.stdout or ""
            raw_stderr = e.stderr or ""
            stdout_str = (
                raw_stdout.decode("utf-8", errors="replace")
                if isinstance(raw_stdout, bytes)
                else raw_stdout
            )
            stderr_str = (
                raw_stderr.decode("utf-8", errors="replace")
                if isinstance(raw_stderr, bytes)
                else raw_stderr
            )
            stdout, t1 = _truncate(stdout_str, config.output_limit_kb)
            stderr, t2 = _truncate(stderr_str, config.output_limit_kb)
            return SandboxResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                truncated=(t1 or t2),
            )
