"""Test MCP server whose ``busy`` tool blocks the server synchronously, like a CPU-bound
decoder in chatqec-mcp-tools (its call_tool handler runs the tool without yielding)."""

import asyncio
import json
import time

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("busy-test")


@app.list_tools()
async def list_tools() -> list[Tool]:
    schema = {"type": "object", "properties": {}}
    return [
        Tool(name="busy", description="blocks for 30 s", inputSchema=schema),
        Tool(name="ping", description="returns at once", inputSchema=schema),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "busy":
        time.sleep(30)  # deliberately blocking: the server cannot serve anything else
    return [TextContent(type="text", text=json.dumps({"tool": name}))]


async def main() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
