"""A wedged MCP server must fail the call, not hang the answer.

``_submit`` used to call ``.result()`` with no timeout. "Never raises" is not
"never hangs", and since the wiki's ``search_pages`` moved onto every query's
critical path a stuck subprocess would block a publicly deployable app forever.

These tests drive ``_submit``/``call_tool`` against a hand-built client (no
subprocess), so they run everywhere rather than being skipped like the
integration tests in ``test_mcp_stdio_client.py``.
"""

import asyncio
import threading

import pytest

from chatqec.conversation.mcp_stdio import StdioMCPClient
from chatqec.retrieval.wiki_search import wiki_chunks


class _SleepySession:
    """Stands in for ClientSession: every call takes far longer than the budget."""

    def __init__(self, delay=30.0):
        self.delay = delay
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        await asyncio.sleep(self.delay)
        raise AssertionError("should have timed out long before this")


@pytest.fixture
def client():
    """A StdioMCPClient with a live worker loop but no spawned server."""
    c = object.__new__(StdioMCPClient)
    c._loop = asyncio.new_event_loop()
    c._thread = threading.Thread(target=c._loop.run_forever, daemon=True)
    c._thread.start()
    c.call_timeout = 0.1
    c._tool_timeouts = {}
    c._session = _SleepySession()
    c._tools = []
    c._restart = lambda: None  # no subprocess here; restart is tested in test_mcp_stdio_restart.py
    yield c
    c._loop.call_soon_threadsafe(c._loop.stop)
    c._thread.join(timeout=5)
    c._loop.close()  # otherwise the loop's __del__ raises a ResourceWarning


def test_submit_raises_instead_of_blocking_forever(client):
    with pytest.raises(TimeoutError):
        client._submit(asyncio.sleep(30), timeout=0.05)


def test_a_timed_out_call_is_cancelled_so_it_stops_occupying_the_loop(client):
    coro = asyncio.sleep(30)
    with pytest.raises(TimeoutError):
        client._submit(coro, timeout=0.05)
    # The loop is still usable for the next call rather than wedged on the old one.
    assert client._submit(asyncio.sleep(0, result="alive"), timeout=5) == "alive"


def test_call_tool_is_bounded_by_the_default_call_timeout(client):
    with pytest.raises(TimeoutError):
        client.call_tool("stim_simulate", {})
    assert client._session.calls == [("stim_simulate", {})]


def test_a_per_tool_timeout_overrides_the_default(client):
    """One server can host both a latency-critical tool and a deliberately slow one
    (``check_freshness`` does network I/O), so the budget has to be per tool."""
    client.call_timeout = 30.0
    client._tool_timeouts = {"search_pages": 0.05}
    with pytest.raises(TimeoutError):
        client.call_tool("search_pages", {"query": "surface code"})


def test_the_timeout_message_names_the_budget(client):
    with pytest.raises(TimeoutError, match=r"0\.05"):
        client._submit(asyncio.sleep(30), timeout=0.05)


def test_a_timing_out_client_degrades_the_answer_to_the_corpus_alone(client):
    """End to end for the failure that matters: wiki_chunks swallows the timeout and
    returns [], so retrieval proceeds with Qdrant chunks only."""
    client._tool_timeouts = {"search_pages": 0.05}
    assert wiki_chunks(client, "surface code threshold", k=8) == []
