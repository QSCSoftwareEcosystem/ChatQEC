"""Fan one MCPAdapter out across several MCP servers.

ChatQEC used to build exactly one StdioMCPClient, so adding the wiki server would
have cost the Stim tools. MCPAdapter depends only on the MCPClient Protocol
(list_tools/call_tool), so a client that fans out satisfies it with no change to
the adapter, the synthesizer, or the tool loop.

A server that fails to start is skipped rather than fatal: losing the wiki should
degrade answers, not break Stim.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from chatqec.conversation.mcp_adapter import MCPClient, MCPToolDef

logger = logging.getLogger(__name__)


class MultiMCPClient:
    """Presents several MCP servers as one, routing calls by tool name."""

    def __init__(self, clients: Sequence[MCPClient]) -> None:
        self._clients = list(clients)
        self._tools: list[MCPToolDef] = []
        self._owner: dict[str, MCPClient] = {}
        for client in self._clients:
            try:
                tools = client.list_tools()
            except Exception:  # a dead server must not take the others down
                logger.warning("MCP server failed to list tools; skipping", exc_info=True)
                continue
            for tool in tools:
                if tool.name in self._owner:
                    logger.warning("Duplicate MCP tool %r; keeping the first", tool.name)
                    continue
                self._owner[tool.name] = client
                self._tools.append(tool)

    def list_tools(self) -> list[MCPToolDef]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        client = self._owner.get(name)
        if client is None:
            raise KeyError(name)
        return client.call_tool(name, args)
