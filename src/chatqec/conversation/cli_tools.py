"""Run the QEC tools as one subprocess per call through the ``chatqec-tools`` CLI.

Satisfies the ``MCPClient`` protocol, so ``MultiMCPClient``, ``MCPAdapter`` and the
tool loop use it unchanged. Compared with ``StdioMCPClient`` there is no long-lived
session: each call starts ``chatqec-tools call <name>``, writes the arguments as JSON
on stdin and reads one JSON object from stdout. A timeout kills that call's process
group, so an abandoned computation cannot hold up later calls, and a native crash in
a vendor decoder ends only that call.

The tools keep their own environment (installed with ``uv tool install``), so heavy
extras such as CUDA-Q never enter the app's environment.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
from typing import Any

from chatqec.conversation.mcp_adapter import MCPToolDef
from chatqec.trace import trace

logger = logging.getLogger(__name__)

TOOLS_CLI = "chatqec-tools"


def is_tools_cli(command: str) -> bool:
    """True when ``command`` launches the tools CLI rather than an MCP server."""
    argv = shlex.split(command)
    return bool(argv) and os.path.basename(argv[0]) == TOOLS_CLI


class CliToolClient:
    """Lists tools once, then runs each call in a fresh ``chatqec-tools`` process."""

    def __init__(
        self,
        *,
        command: str = TOOLS_CLI,
        list_timeout: float = 30.0,
        call_timeout: float = 60.0,
        tool_timeouts: dict[str, float] | None = None,
    ) -> None:
        """Run ``<command> list`` once; raise if the CLI is missing or broken.

        ``call_timeout`` bounds every call and ``tool_timeouts`` overrides it per tool,
        with the same meaning as in ``StdioMCPClient``.
        """
        self.server_command = command  # MultiMCPClient labels log lines with this
        self._argv = shlex.split(command)
        self.call_timeout = call_timeout
        self._tool_timeouts = dict(tool_timeouts or {})
        code, out, err = self._run(["list"], "", list_timeout)
        if code != 0:
            raise RuntimeError(f"{command!r} list failed (exit {code}): {err.strip()[-500:]}")
        self._tools = [
            MCPToolDef(
                name=t["name"],
                description=t.get("description") or "",
                input_schema=t.get("input_schema") or {},
            )
            for t in json.loads(out)
        ]
        trace("tools_cli.tools_loaded", command=command, tools=[t.name for t in self._tools])

    def _run(self, args: list[str], stdin: str, timeout: float) -> tuple[int, str, str]:
        """Run the CLI; on timeout kill its whole process group and raise TimeoutError."""
        proc = subprocess.Popen(  # operator-configured command, no shell
            [*self._argv, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group, so the kill reaches children
        )
        try:
            out, err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise TimeoutError(f"tool call exceeded {timeout:g}s") from None
        return proc.returncode, out, err

    def list_tools(self) -> list[MCPToolDef]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        trace("tools_cli.call", name=name, args=args)
        timeout = self._tool_timeouts.get(name, self.call_timeout)
        try:
            code, out, err = self._run(["call", name], json.dumps(args), timeout)
        except TimeoutError as exc:
            # Returned, not raised: the model can tell the user the computation was too
            # slow and suggest fewer shots, instead of the whole answer failing.
            failed = {"ok": False, "error_message": f"{exc}; try fewer shots or a smaller code"}
            trace("tools_cli.timeout", name=name, timeout=timeout)
            return failed
        if err.strip():
            logger.debug("%s %s stderr: %s", self.server_command, name, err.strip()[-2000:])
        if code != 0:
            failed = {
                "ok": False,
                "error_message": f"tool process exited with code {code}: {err.strip()[-500:]}",
            }
            trace("tools_cli.call_error", name=name, output=failed)
            return failed
        try:
            parsed: Any = json.loads(out)
        except json.JSONDecodeError:
            failed = {"ok": False, "error_message": f"tool returned non-JSON output: {out[:500]}"}
            trace("tools_cli.call_error", name=name, output=failed)
            return failed
        result: dict[str, Any] = (
            parsed
            if isinstance(parsed, dict)
            else {"ok": False, "error_message": f"tool returned {type(parsed).__name__}"}
        )
        trace("tools_cli.call_result", name=name, output=result)
        return result

    def close(self) -> None:
        """Nothing to release: no process outlives its call."""
