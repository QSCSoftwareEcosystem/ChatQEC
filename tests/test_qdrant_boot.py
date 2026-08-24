"""The Streamlit app brings its own Qdrant up when one isn't already running."""

from pathlib import Path

import pytest

from chatqec_app.qdrant_boot import QdrantBootError, ensure_qdrant

_URL = "http://localhost:6333"


def _never_spawn(*_a, **_k):  # pragma: no cover - failure guard
    raise AssertionError("spawn must not be called")


def test_returns_early_when_qdrant_already_answers():
    assert ensure_qdrant(url=_URL, probe=lambda _u: True, spawn=_never_spawn) == "already-running"


def test_disabled_by_env_never_probes_or_spawns(monkeypatch):
    monkeypatch.setenv("CHATQEC_AUTOSTART_QDRANT", "0")
    assert ensure_qdrant(url=_URL, probe=_never_spawn, spawn=_never_spawn) == "disabled"


def test_remote_url_is_never_autostarted():
    with pytest.raises(QdrantBootError, match="not local"):
        ensure_qdrant(
            url="http://qdrant.example.com:6333", probe=lambda _u: False, spawn=_never_spawn
        )


def test_missing_binary_raises_with_the_command_to_run():
    with pytest.raises(QdrantBootError, match="qdrant_data/storage"):
        ensure_qdrant(url=_URL, probe=lambda _u: False, spawn=_never_spawn, binary=None)


def test_spawns_then_waits_until_ready(tmp_path):
    calls: list[Path] = []
    states = iter([False, False, True])

    def spawn(**kw):
        calls.append(kw["storage_path"])

    status = ensure_qdrant(
        url=_URL,
        repo_root=tmp_path,
        probe=lambda _u: next(states),
        spawn=spawn,
        binary="/usr/bin/qdrant",
        sleep=lambda _s: None,
    )
    assert status == "started"
    assert calls == [tmp_path / "qdrant_data" / "storage"]


def test_spawn_that_never_becomes_ready_raises_pointing_at_the_log(tmp_path):
    with pytest.raises(QdrantBootError, match=r"qdrant\.log"):
        ensure_qdrant(
            url=_URL,
            repo_root=tmp_path,
            probe=lambda _u: False,
            spawn=lambda **kw: None,
            binary="/usr/bin/qdrant",
            sleep=lambda _s: None,
            timeout=0.05,
        )


def test_build_bot_ensures_qdrant_before_building(monkeypatch):
    pytest.importorskip("streamlit")
    order: list[str] = []
    monkeypatch.setattr(
        "chatqec_app.utils.ensure_qdrant", lambda: order.append("qdrant") or "already-running"
    )
    monkeypatch.setattr(
        "chatqec_app.utils._build_chatqec", lambda **_kw: order.append("bot") or "bot"
    )

    from chatqec_app.utils import _build_bot

    _build_bot()
    assert order == ["qdrant", "bot"]
