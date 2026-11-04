"""Smoke tests for the Streamlit pages using AppTest.

We monkeypatch the CLI builder (the actual factory used by chatqec_app.utils) so
the pages don't try to reach a live Anthropic/Qdrant during the test.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _stub_chatqec(monkeypatch):
    from chatqec.api import ChatQEC

    stub = MagicMock(spec=ChatQEC)
    stub.ask_stream.return_value = iter([])
    stub.ask.return_value = MagicMock(text="ok", citations=[], tool_calls=[], confidence="low")
    # Patch the symbol that chatqec_app.utils.get_chatqec() actually calls.
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: stub)
    # Belt-and-suspenders: also patch the helper in case @st.cache_resource is
    # involved across tests.
    monkeypatch.setattr("chatqec_app.utils.get_chatqec", lambda: stub)


def test_streamlit_root_renders():
    at = AppTest.from_file(str(REPO_ROOT / "chatqec_app/streamlit_app.py"), default_timeout=10)
    at.run()
    assert not at.exception
    # streamlit_app.py redirects straight to the Chat page (st.switch_page).
    # Chat has no st.title any more (removed in 32d5a39); its chat input is
    # the element that proves the redirect landed.
    assert len(at.chat_input) == 1


def test_about_page_renders():
    at = AppTest.from_file(str(REPO_ROOT / "chatqec_app/pages/4_About.py"), default_timeout=10)
    at.run()
    assert not at.exception


def test_watchlist_page_renders_when_yaml_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "watchlist.yaml").write_text(
        "arxiv:\n  categories: [quant-ph]\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(
        str(REPO_ROOT / "chatqec_app/pages/3_Watchlist.py"),
        default_timeout=10,
    )
    at.run()
    assert not at.exception


def test_sources_page_renders_when_empty_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "registry.jsonl").touch()
    at = AppTest.from_file(
        str(REPO_ROOT / "chatqec_app/pages/2_Sources.py"),
        default_timeout=10,
    )
    at.run()
    assert not at.exception


def test_chat_page_passes_a_per_session_memory(monkeypatch):
    """The cached ChatQEC is shared by every visitor; the Chat page must send
    each session's own ConversationMemory so history never crosses sessions."""
    from chatqec.api import ChatQEC
    from chatqec.conversation.memory import ConversationMemory

    stub = MagicMock(spec=ChatQEC)
    stub.ask_stream.return_value = iter([])
    stub.ask.return_value = MagicMock(text="ok", citations=[], tool_calls=[], confidence="low")
    monkeypatch.setattr("chatqec.cli._build_chatqec", lambda **_: stub)
    monkeypatch.setattr("chatqec_app.utils.get_chatqec", lambda: stub)

    at = AppTest.from_file(str(REPO_ROOT / "chatqec_app/pages/1_Chat.py"), default_timeout=20)
    at.run()
    at.chat_input[0].set_value("What is the surface code?").run()

    calls = stub.ask.call_args_list + stub.ask_stream.call_args_list
    assert calls, "the Chat page never called the bot"
    assert all(isinstance(c.kwargs.get("memory"), ConversationMemory) for c in calls)
