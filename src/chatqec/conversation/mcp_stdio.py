"""Real stdio-based MCP client satisfying the MCPClient protocol.

Wraps the official ``mcp`` async ``ClientSession`` (stdio transport) behind the
synchronous ``list_tools()`` / ``call_tool()`` interface the rest of ChatQEC
expects. A single long-lived worker coroutine owns the async context on a
dedicated event-loop thread — it both enters and exits the ``async with``
blocks, avoiding anyio "cancel scope in a different task" errors. Per-call
coroutines (list/call) are submitted onto that loop via
``run_coroutine_threadsafe``.

Every submitted call is bounded by a timeout. "Never raises" is not "never
hangs": a wedged server subprocess would otherwise block the answering thread
forever, and since Task 9 the wiki's ``search_pages`` sits on every query's
critical path, not just on tool-calling turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from chatqec.conversation.mcp_adapter import MCPToolDef
from chatqec.trace import trace


class StdioMCPClient:
    """Spawns the MCP server as a subprocess and talks to it over stdio."""

    def __init__(
        self,
        *,
        server_command: str,
        ready_timeout: float = 30.0,
        call_timeout: float = 60.0,
        tool_timeouts: dict[str, float] | None = None,
    ) -> None:
        """Spawn the server.

        ``call_timeout`` bounds every tool call; ``tool_timeouts`` overrides it per
        tool name. The override exists because a single server can host both a
        latency-critical tool (the wiki's ``search_pages``, run on every query) and
        a deliberately slow one (``check_freshness`` does network I/O), so one
        number for the whole server would be wrong for one of them.
        """
        argv = shlex.split(server_command)
        trace("mcp.spawn", server_command=server_command, argv=argv)
        self.server_command = server_command
        self.call_timeout = call_timeout
        self._tool_timeouts = dict(tool_timeouts or {})
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._error: BaseException | None = None

        self._worker_fut = asyncio.run_coroutine_threadsafe(self._worker(argv), self._loop)
        if not self._ready.wait(timeout=ready_timeout):
            raise RuntimeError(f"MCP server {argv!r} did not become ready in {ready_timeout}s")
        if self._error is not None:
            raise self._error
        self._tools: list[MCPToolDef] = self._load_tools()

    async def _worker(self, argv: list[str]) -> None:
        self._stop = asyncio.Event()
        # StdioServerParameters defaults to a minimal env (HOME/LOGNAME/PATH/SHELL/
        # TERM/USER only, per mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS) unless
        # env= is passed explicitly -- without this, secrets like QISKIT_IBM_TOKEN
        # set in the parent process (e.g. via .env + load_dotenv()) never reach the
        # spawned server subprocess, which then reports missing credentials even
        # though they're set. Pass the full parent environment: this subprocess
        # command is operator-configured (CHATQEC_MCP_SERVER), not a third-party
        # server the MCP stdio spec's minimal-env default is meant to sandbox.
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=os.environ.copy())
        try:
            # Nested (not combined): ClientSession needs the outer context's streams.
            async with stdio_client(params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # surfaced to the __init__ caller
            self._error = exc
            self._ready.set()

    def _submit(self, coro: Any, *, timeout: float | None = None) -> Any:
        """Run ``coro`` on the worker loop and wait, raising TimeoutError if it stalls."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            # Cancel so a wedged call stops occupying the loop after we give up.
            future.cancel()
            raise TimeoutError(f"MCP call exceeded {timeout}s") from None

    def _load_tools(self) -> list[MCPToolDef]:
        assert self._session is not None
        resp = self._submit(self._session.list_tools(), timeout=self.call_timeout)
        tools = [
            MCPToolDef(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in resp.tools
        ]
        trace("mcp.tools_loaded", tools=[t.name for t in tools])
        return tools

    def list_tools(self) -> list[MCPToolDef]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self._session is not None
        trace("mcp.call_tool", name=name, args=args)
        timeout = self._tool_timeouts.get(name, self.call_timeout)
        result = self._submit(self._session.call_tool(name, args), timeout=timeout)
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parsed: dict[str, Any] = json.loads(text)
                trace("mcp.call_result", name=name, output=parsed)
                return parsed
        trace("mcp.call_result", name=name, output={})
        return {}

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        with contextlib.suppress(Exception):  # best-effort shutdown
            self._worker_fut.result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()
