"""``_mcp_client_or_none`` keeps an all-failed --mcp-server list from looking wired.

``MultiMCPClient([])`` is a valid, non-None object with an empty tool list — so the
CLI's MCP wiring must decide "usable or not" by looking at the started ``clients``,
not by testing the constructed client for ``is not None``. This guards against a
regression where a typo'd/dead ``--mcp-server`` command silently leaves
``synthesizer.mcp_adapter`` and the wiki searcher looking configured (which then
suppresses generic corpus figures and burns a doomed ``wiki_chunks`` call per query)
instead of falling back cleanly to "no MCP tools at all", matching pre-fan-out
behaviour.
"""

from chatqec.cli import _mcp_client_or_none
from chatqec.conversation.mcp_adapter import MCPToolDef


class FakeClient:
    def __init__(self, names):
        self._names = names

    def list_tools(self):
        return [MCPToolDef(name=n, description=f"d:{n}", input_schema={}) for n in self._names]

    def call_tool(self, name, args):
        raise KeyError(name)


def test_no_started_clients_yields_none():
    """The regression case: every --mcp-server command failed to start.

    Before this guard, ``MultiMCPClient([])`` — a valid, non-None object whose
    ``list_tools()`` is simply empty — was built unconditionally, so downstream
    ``is not None`` checks (mcp_adapter, the wiki searcher) read "configured"
    with nothing actually usable behind them.
    """
    assert _mcp_client_or_none([]) is None


def test_at_least_one_started_client_yields_a_usable_client():
    client = _mcp_client_or_none([FakeClient(["search_pages"])])
    assert client is not None
    assert [t.name for t in client.list_tools()] == ["search_pages"]
