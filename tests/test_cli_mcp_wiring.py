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


# --------------------------------------------------------------------------- #
# the wiki gate: "is search_pages reachable", not "did some MCP server start"
# --------------------------------------------------------------------------- #

from chatqec.cli import _wiki_is_available  # noqa: E402

_ON = {"wiki": {"enabled": True}}


def test_a_client_without_search_pages_leaves_the_wiki_off():
    """The regression this guards: CHATQEC_MCP_SERVER="chatqec-mcp-tools" with no
    wiki server is the documented default. Gating on `_mcp_client is not None` sent
    a doomed call per question, and MultiMCPClient's KeyError was logged with a
    full traceback each time."""
    assert _wiki_is_available(_ON, FakeClient(["stim_simulate", "stim_diagram"])) is False


def test_a_client_that_advertises_search_pages_turns_the_wiki_on():
    assert _wiki_is_available(_ON, FakeClient(["stim_simulate", "search_pages"])) is True


def test_no_client_leaves_the_wiki_off():
    assert _wiki_is_available(_ON, None) is False


def test_config_disabled_wins_over_an_available_tool():
    assert _wiki_is_available({"wiki": {"enabled": False}}, FakeClient(["search_pages"])) is False
    assert _wiki_is_available({}, FakeClient(["search_pages"])) is False


def test_a_client_that_cannot_be_listed_leaves_the_wiki_off():
    class Broken:
        def list_tools(self):
            raise RuntimeError("server died")

    assert _wiki_is_available(_ON, Broken()) is False


def test_mcp_call_timeout_comes_from_config_or_the_env_override(monkeypatch):
    """Benchmark runs use slow decoders (Relay-BP, MWPF), so they raise the MCP call
    timeout with CHATQEC_MCP_TIMEOUT_SECONDS; the config value stays the default."""
    from chatqec.conversation import mcp_stdio
    from chatqec.cli import _start_mcp_clients

    seen = []

    class _Fake:
        def __init__(self, **kwargs):
            seen.append(kwargs["call_timeout"])

    monkeypatch.setattr(mcp_stdio, "StdioMCPClient", _Fake)
    cfg = {"mcp": {"request_timeout_seconds": 60}}
    monkeypatch.delenv("CHATQEC_MCP_TIMEOUT_SECONDS", raising=False)
    _start_mcp_clients(["tools"], cfg)
    monkeypatch.setenv("CHATQEC_MCP_TIMEOUT_SECONDS", "1800")
    _start_mcp_clients(["tools"], cfg)
    assert seen == [60.0, 1800.0]
