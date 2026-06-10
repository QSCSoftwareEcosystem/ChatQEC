"""Stdio-based MCP client wrapper that satisfies the MCPClient protocol.

This is a placeholder for v1: the chatqec-mcp-tools server uses MCP's async
stdio handshake, and a full async client lands as a follow-up patch. For now
this stub exists so the CLI ``--mcp-server`` option type-checks and so the
adapter API is unchanged. Real users wire the MCP server through Claude
Desktop's standard config; v2 will add a proper async stdio implementation.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from chatqec.conversation.mcp_adapter import MCPToolDef


class StdioMCPClient:
    """Spawns the MCP server as a subprocess and talks to it over stdio."""

    def __init__(self, *, server_command: str) -> None:
        argv = shlex.split(server_command)
        self.proc = subprocess.Popen(  # pragma: no cover
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def list_tools(self) -> list[MCPToolDef]:  # pragma: no cover
        return []

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        return {}
