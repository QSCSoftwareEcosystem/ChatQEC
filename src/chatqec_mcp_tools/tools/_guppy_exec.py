"""Run a Python program in an isolated child process (the runner behind guppy_run).

Not an OS sandbox: there is no filesystem or network isolation, because unprivileged user
namespaces (bwrap/unshare) are blocked on many machines (e.g. Ubuntu's AppArmor policy), so the
guard against hostile code is the source gate in _guppy_gate.py. What this adds:

  - ``python -I`` (isolated mode), a fresh temporary directory as cwd and HOME, and a minimal
    environment, so no API keys or tokens are inherited
  - a new session, so a timeout kills the whole process group (Selene starts a Zig toolchain)
  - CPU-time, file-size and address-space limits, and truncated output

It deliberately does NOT set RLIMIT_NPROC as ``sandbox.py`` does: that limit counts every process
and thread of the whole user, so on a busy account it makes ordinary imports (numpy starts
threads) fail while protecting nothing.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SAFE_PATH = "/usr/bin:/bin"


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


# Runs inside the child before the program: applies the limits, then executes prog.py as __main__.
# Done here rather than with preexec_fn, which Python documents as unsafe in threaded programs
# (the MCP server is threaded).
_BOOTSTRAP = """
import resource, runpy, sys
cpu, mem, fsize = (int(v) for v in sys.argv[2:5])
mb = 2 ** 20
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize * mb, fsize * mb))
resource.setrlimit(resource.RLIMIT_AS, (mem * mb, mem * mb))
program = sys.argv[1]
sys.argv = [program]
runpy.run_path(program, run_name="__main__")
"""


def _truncate(text: str, limit_kb: int) -> tuple[str, bool]:
    limit = limit_kb * 1024
    return (text, False) if len(text) <= limit else (text[:limit] + "\n... [truncated]", True)


def _text(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def run_isolated(
    code: str,
    *,
    wall_seconds: float,
    cpu_seconds: int,
    memory_mb: int,
    fsize_mb: int = 50,
    output_limit_kb: int = 100,
) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="chatqec-guppy-") as workdir:
        (Path(workdir) / "prog.py").write_text(code, encoding="utf-8")
        env = {
            "HOME": workdir,
            "TMPDIR": workdir,
            "PATH": _SAFE_PATH,
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
        }
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _BOOTSTRAP,
                "prog.py",
                str(cpu_seconds),
                str(memory_mb),
                str(fsize_mb),
            ],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=wall_seconds)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial_out, partial_err = _text(exc.stdout), _text(exc.stderr)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            out, err = partial_out + out, partial_err + err
        finally:
            # a finished or killed child may still have grandchildren in its group
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        out, t_out = _truncate(out, output_limit_kb)
        err, t_err = _truncate(err, output_limit_kb)
        return ExecResult(
            exit_code=-1 if timed_out else proc.returncode,
            stdout=out,
            stderr=err,
            timed_out=timed_out,
            truncated=t_out or t_err,
        )
