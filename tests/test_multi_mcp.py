"""One adapter, several MCP servers: the Stim sibling and the wiki at once."""

import pytest

from chatqec.conversation.mcp_adapter import MCPToolDef
from chatqec.conversation.multi_mcp import MultiMCPClient


class FakeClient:
    def __init__(self, names, result=None, raises=None):
        self._names = names
        self._result = result or {"ok": True}
        self._raises = raises
        self.calls = []

    def list_tools(self):
        if self._raises:
            raise self._raises
        return [MCPToolDef(name=n, description=f"d:{n}", input_schema={}) for n in self._names]

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._result


def test_lists_the_union_of_all_servers_tools():
    c = MultiMCPClient([FakeClient(["stim_simulate"]), FakeClient(["search_pages"])])
    assert [t.name for t in c.list_tools()] == ["stim_simulate", "search_pages"]


def test_routes_each_call_to_the_server_that_owns_the_tool():
    a, b = FakeClient(["stim_simulate"]), FakeClient(["search_pages"], result={"rows": []})
    c = MultiMCPClient([a, b])
    assert c.call_tool("search_pages", {"query": "x"}) == {"rows": []}
    assert b.calls == [("search_pages", {"query": "x"})]
    assert a.calls == []


def test_first_server_wins_a_name_collision():
    a, b = FakeClient(["dup"], result={"from": "a"}), FakeClient(["dup"], result={"from": "b"})
    c = MultiMCPClient([a, b])
    assert [t.name for t in c.list_tools()] == ["dup"]
    assert c.call_tool("dup", {}) == {"from": "a"}


def test_a_server_that_fails_to_list_is_skipped_not_fatal():
    broken = FakeClient(["never"], raises=RuntimeError("server died"))
    good = FakeClient(["stim_simulate"])
    c = MultiMCPClient([broken, good])
    assert [t.name for t in c.list_tools()] == ["stim_simulate"]


def test_unknown_tool_raises_keyerror():
    c = MultiMCPClient([FakeClient(["a"])])
    with pytest.raises(KeyError):
        c.call_tool("nope", {})


def test_no_clients_is_harmless():
    assert MultiMCPClient([]).list_tools() == []
