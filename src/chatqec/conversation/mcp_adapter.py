"""Adapter: MCP tool definitions <-> Anthropic tool schemas.

The adapter is given an MCP client (real or fake). It exposes:
    - tool_schemas() — list of AnthropicToolSchema for tests
    - tool_schemas_for_sdk() — the list[dict] Anthropic SDK expects
    - invoke(name, args) — proxy a tool call to the MCP client
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MCPToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class AnthropicToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient(Protocol):
    def list_tools(self) -> list[MCPToolDef]: ...
    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class MCPAdapter:
    def __init__(self, *, client: MCPClient) -> None:
        self.client = client
        self._tools: list[MCPToolDef] = list(client.list_tools())

    def tool_schemas(self) -> list[AnthropicToolSchema]:
        return [
            AnthropicToolSchema(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in self._tools
        ]

    def tool_schemas_for_sdk(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools
        ]

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        for t in self._tools:
            if t.name == name:
                return self.client.call_tool(name, args)
        raise KeyError(name)
