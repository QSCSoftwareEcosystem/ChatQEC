"""The Streamlit app wires the MCP/Stim server from CHATQEC_MCP_SERVER."""

import pytest

pytest.importorskip("streamlit")


def test_build_bot_passes_mcp_server_from_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "chatqec_app.utils._build_chatqec", lambda **kw: captured.update(kw) or "bot"
    )
    monkeypatch.setenv("CHATQEC_MCP_SERVER", "chatqec-mcp-tools")

    from chatqec_app.utils import _build_bot

    _build_bot()
    assert captured["mcp_server"] == "chatqec-mcp-tools"


def test_build_bot_no_mcp_server_when_env_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "chatqec_app.utils._build_chatqec", lambda **kw: captured.update(kw) or "bot"
    )
    monkeypatch.delenv("CHATQEC_MCP_SERVER", raising=False)

    from chatqec_app.utils import _build_bot

    _build_bot()
    assert captured["mcp_server"] is None


def test_build_bot_loads_dotenv(monkeypatch):
    # Streamlit doesn't go through the CLI's main(), so the app must load .env
    # itself or provider keys (GEMINI_API_KEY, ...) are missing.
    calls = []
    monkeypatch.setattr(
        "chatqec_app.utils.load_dotenv", lambda *a, **k: calls.append(True), raising=False
    )
    monkeypatch.setattr("chatqec_app.utils._build_chatqec", lambda **kw: "bot")

    from chatqec_app.utils import _build_bot

    _build_bot()
    assert calls == [True]
