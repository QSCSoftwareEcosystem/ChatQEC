from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield tmp_path


def test_cache_miss_calls_fetch_fn_and_persists():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    calls = []

    def fetch():
        calls.append(1)
        return {"value": 42}

    result = get_or_fetch("k1", ttl_seconds=60, fetch_fn=fetch)

    assert result.payload == {"value": 42}
    assert result.age_seconds == 0.0
    assert result.stale is False
    assert len(calls) == 1

    result2 = get_or_fetch("k1", ttl_seconds=60, fetch_fn=fetch)
    assert result2.payload == {"value": 42}
    assert len(calls) == 1  # second call served from cache, fetch_fn not called again


def test_cache_expiry_triggers_refetch():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    get_or_fetch("k2", ttl_seconds=0, fetch_fn=fetch)
    time.sleep(0.01)
    result = get_or_fetch("k2", ttl_seconds=0, fetch_fn=fetch)

    assert len(calls) == 2
    assert result.payload == {"n": 2}


def test_corrupt_cache_file_is_treated_as_miss(tmp_path):
    from chatqec_mcp_tools.tools.ibm_hardware._cache import _cache_path, get_or_fetch

    path = _cache_path("k3")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{{", encoding="utf-8")

    result = get_or_fetch("k3", ttl_seconds=60, fetch_fn=lambda: {"ok": True})

    assert result.payload == {"ok": True}
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == {"ok": True}


def test_stale_cache_served_when_fetch_fails():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    get_or_fetch("k4", ttl_seconds=0, fetch_fn=lambda: {"v": "first"})

    def failing_fetch():
        raise RuntimeError("network down")

    result = get_or_fetch("k4", ttl_seconds=0, fetch_fn=failing_fetch)

    assert result.payload == {"v": "first"}
    assert result.stale is True


def test_fetch_failure_with_no_cache_raises():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    def failing_fetch():
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        get_or_fetch("k5", ttl_seconds=60, fetch_fn=failing_fetch)


def test_credentials_error_is_never_masked_by_stale_cache():
    """A broken/missing credential isn't a transient blip -- serving a days-old
    cached snapshot in its place would silently look like live data forever.
    IBMCredentialsError must propagate even when a (possibly very old) cached
    payload exists, unlike every other fetch failure."""
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
    from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError

    get_or_fetch("k6", ttl_seconds=0, fetch_fn=lambda: {"v": "first"})

    def failing_fetch():
        raise IBMCredentialsError("no token configured")

    with pytest.raises(IBMCredentialsError, match="no token configured"):
        get_or_fetch("k6", ttl_seconds=0, fetch_fn=failing_fetch)


def test_keys_are_isolated():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    get_or_fetch("backend/a", ttl_seconds=60, fetch_fn=lambda: {"name": "a"})
    result = get_or_fetch("backend/b", ttl_seconds=60, fetch_fn=lambda: {"name": "b"})

    assert result.payload == {"name": "b"}
