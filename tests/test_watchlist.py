"""Watchlist YAML parsing tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chatqec.ingestion.watchlist import Watchlist, load_watchlist


SAMPLE = """
arxiv:
  categories: [quant-ph]
  keywords: [surface code, qLDPC]
  authors: [Preskill, Gottesman]
github:
  - repo: quantumlib/Stim
  - repo: oscarhiggott/PyMatching
youtube:
  channels: ["@GoogleQuantumAI", "@qiskit"]
rss:
  - https://quantum-journal.org/feed/
"""


def test_load_watchlist_parses_all_sections(tmp_path: Path):
    p = tmp_path / "watchlist.yaml"
    p.write_text(SAMPLE)
    wl = load_watchlist(p)
    assert isinstance(wl, Watchlist)
    assert wl.arxiv.categories == ["quant-ph"]
    assert "surface code" in wl.arxiv.keywords
    assert wl.github[0].repo == "quantumlib/Stim"
    assert "@GoogleQuantumAI" in wl.youtube.channels
    assert str(wl.rss[0]).startswith("https://quantum-journal.org")


def test_load_watchlist_rejects_invalid_github_repo(tmp_path: Path):
    bad = "github:\n  - repo: just-one-segment\n"
    p = tmp_path / "watchlist.yaml"
    p.write_text(bad)
    with pytest.raises(ValidationError):
        load_watchlist(p)


def test_load_watchlist_allows_missing_sections(tmp_path: Path):
    p = tmp_path / "watchlist.yaml"
    p.write_text("arxiv:\n  categories: []\n")
    wl = load_watchlist(p)
    assert wl.github == []
    assert wl.youtube.channels == []
    assert wl.rss == []
