"""A timed-out tool call must not poison the calls after it.

chatqec-mcp-tools runs a tool inside its call_tool handler without yielding, so when the
client gave up on a slow decoder call the server kept computing it, and every later call
in the session queued behind the abandoned one and timed out too (a benchmark run lost a
surface-code item, which takes seconds, to a 30-minute timeout this way). After a timeout
the client must replace the server process, which also stops the abandoned work.
"""

import shlex
import sys
import time
from pathlib import Path

import pytest

from chatqec.conversation.mcp_stdio import StdioMCPClient

SERVER = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).with_name('_mcp_busy_server.py')))}"


@pytest.fixture
def client():
    c = StdioMCPClient(server_command=SERVER, call_timeout=1.0)
    yield c
    c.close()


def test_a_call_after_a_timeout_is_not_queued_behind_the_abandoned_one(client):
    with pytest.raises(TimeoutError):
        client.call_tool("busy", {})
    t0 = time.monotonic()
    assert client.call_tool("ping", {}) == {"tool": "ping"}
    assert time.monotonic() - t0 < 10  # without a restart this waits ~29 s and times out


def test_the_abandoned_server_process_is_gone_after_a_timeout(client):
    old = client.server_pid
    with pytest.raises(TimeoutError):
        client.call_tool("busy", {})
    assert client.server_pid != old
    assert (
        not Path(f"/proc/{old}").exists()
        or "zombie" in Path(f"/proc/{old}/status").read_text().lower()
    )
