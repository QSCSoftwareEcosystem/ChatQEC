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
    # streamlit_app.py now redirects straight to the Chat page (st.switch_page)
    assert any("Chat" in t.value for t in at.title)


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
