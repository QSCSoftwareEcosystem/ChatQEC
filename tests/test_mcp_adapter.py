"""MCP adapter tests — uses an in-process FakeMCPClient instead of stdio."""

import pytest

from chatqec.conversation.mcp_adapter import (
    AnthropicToolSchema,
    MCPAdapter,
    MCPToolDef,
)


class _FakeMCPClient:
    def __init__(self, tools, results):
        self._tools = tools
        self._results = results
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return self._tools

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._results[name]


def test_adapter_converts_mcp_tools_to_anthropic_schemas():
    fake = _FakeMCPClient(
        tools=[
            MCPToolDef(
                name="t1",
                description="d1",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            )
        ],
        results={"t1": {"ok": True}},
    )
    a = MCPAdapter(client=fake)
    schemas = a.tool_schemas()
    assert len(schemas) == 1
    assert isinstance(schemas[0], AnthropicToolSchema)
    assert schemas[0].name == "t1"
    assert schemas[0].description == "d1"
    assert schemas[0].input_schema["type"] == "object"


def test_adapter_invokes_proxies_to_client():
    fake = _FakeMCPClient(
        tools=[MCPToolDef(name="t1", description="d", input_schema={})],
        results={"t1": {"ok": True, "summary": "yay"}},
    )
    a = MCPAdapter(client=fake)
    out = a.invoke("t1", {"x": 1})
    assert out == {"ok": True, "summary": "yay"}
    assert fake.calls == [("t1", {"x": 1})]


def test_adapter_raises_for_unknown_tool():
    fake = _FakeMCPClient(tools=[], results={})
    a = MCPAdapter(client=fake)
    with pytest.raises(KeyError):
        a.invoke("nope", {})


def test_adapter_anthropic_tools_payload_matches_sdk_format():
    fake = _FakeMCPClient(
        tools=[
            MCPToolDef(
                name="stim_simulate",
                description="run",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        results={},
    )
    a = MCPAdapter(client=fake)
    sdk_payload = a.tool_schemas_for_sdk()
    assert sdk_payload == [
        {
            "name": "stim_simulate",
            "description": "run",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
