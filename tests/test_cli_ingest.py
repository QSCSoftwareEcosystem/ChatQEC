"""CLI tests for `chatqec ingest` subcommands.

Heavy network calls are not exercised here; we monkeypatch the fetcher's
internal `arxiv.Search` so the tests stay hermetic.
"""

from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner

from chatqec.cli import main


@pytest.fixture
def chunk_dir(tmp_path: Path) -> Path:
    out = tmp_path / "chunks"
    out.mkdir()
    return out


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "registry.jsonl"


def test_ingest_url_writes_chunks(
    tmp_path: Path, chunk_dir: Path, registry_path: Path, monkeypatch
):
    # Stub the HTTP+defuddle pipeline to return canned markdown
    monkeypatch.setattr(
        "chatqec.cli._fetch_url_as_markdown",
        lambda url: ("Surface codes hit a milestone", "Body. " * 500, "2026-01-15"),
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(chunk_dir),
            "--registry",
            str(registry_path),
            "--url",
            "https://example.com/post",
        ],
    )
    assert result.exit_code == 0, result.output

    # A chunk file landed
    files = list(chunk_dir.glob("*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Surface codes" in text

    # Registry got a pending_review record
    reg_text = registry_path.read_text()
    assert "pending_review" in reg_text


def test_ingest_url_empty_extraction_guarded(
    tmp_path: Path, chunk_dir: Path, registry_path: Path, monkeypatch
):
    # Simulate the real bug: `_fetch_url_as_markdown` produced no usable text
    # (e.g. trafilatura silently returned None on already-markdown input).
    monkeypatch.setattr(
        "chatqec.cli._fetch_url_as_markdown",
        lambda url: (url, "", ""),
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(chunk_dir),
            "--registry",
            str(registry_path),
            "--url",
            "https://tsim.mintlify.app/api-reference/circuit.md",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "no chunks produced" in result.output
    assert "extraction empty" in result.output

    # Nothing should have been written on the guarded failure path
    assert list(chunk_dir.glob("*.jsonl")) == []


def test_fetch_url_as_markdown_skips_trafilatura_for_md_suffix(monkeypatch):
    """A `.md` URL is already plain markdown — the raw text must be used as-is
    without running it through trafilatura's HTML-boilerplate remover, which
    can silently return None on non-HTML input (the root cause of the bug)."""
    from chatqec.cli import _fetch_url_as_markdown

    md_content = "# Circuit API\n\nSome plain markdown body, not HTML.\n"

    class FakeResponse:
        text = md_content
        headers: ClassVar[dict[str, str]] = {}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("requests.get", lambda url, timeout=30: FakeResponse())

    def _boom(*args, **kwargs):
        raise AssertionError("trafilatura.extract must not be called for plain markdown input")

    monkeypatch.setattr("trafilatura.extract", _boom)

    title, md_text, _publish_date = _fetch_url_as_markdown(
        "https://tsim.mintlify.app/api-reference/circuit.md"
    )
    assert md_text == md_content
    assert title == "https://tsim.mintlify.app/api-reference/circuit.md"


def test_fetch_url_as_markdown_skips_trafilatura_for_markdown_content_type(monkeypatch):
    """Same skip behavior when the URL has no `.md` suffix but the response
    `Content-Type` header declares plain markdown/text."""
    from chatqec.cli import _fetch_url_as_markdown

    md_content = "# Notes\n\nPlain markdown served without a `.md` suffix.\n"

    class FakeResponse:
        text = md_content
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/markdown; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("requests.get", lambda url, timeout=30: FakeResponse())

    def _boom(*args, **kwargs):
        raise AssertionError("trafilatura.extract must not be called for plain markdown input")

    monkeypatch.setattr("trafilatura.extract", _boom)

    _title, md_text, _publish_date = _fetch_url_as_markdown("https://example.com/raw-notes")
    assert md_text == md_content


def test_ingest_arxiv_with_resolve_by_title(
    monkeypatch, tmp_path: Path, chunk_dir: Path, registry_path: Path
):
    from chatqec.ingestion.resolver import ResolverHit

    monkeypatch.setattr(
        "chatqec.cli._resolve_and_fetch_arxiv",
        lambda title, authors, year: (
            ResolverHit(
                arxiv_id="2408.13687v1",
                title="Quantum error correction below the surface code threshold",
                first_author="Rajeev Acharya",
                year=2024,
            ),
            "Long abstract body. " * 100,
        ),
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(chunk_dir),
            "--registry",
            str(registry_path),
            "--resolve-by-title",
            "below the surface code threshold",
            "--author",
            "Acharya",
            "--year",
            "2024",
        ],
    )
    assert result.exit_code == 0, result.output
    files = list(chunk_dir.glob("arxiv_2408.13687v1*"))
    assert len(files) == 1


def test_watch_refresh_dry_run_prints_candidates(monkeypatch, tmp_path: Path):
    # Stub the watchlist refresh to return a fixed candidate list
    def fake_collect(_wl):
        return [
            ("arxiv:2408.13687v1", "Below threshold", ["Acharya"], 2024),
            ("github:quantumlib/Stim@abc12345", "Stim README", ["Gidney"], 2024),
        ]

    monkeypatch.setattr("chatqec.cli._collect_watchlist_candidates", fake_collect)

    wl_path = tmp_path / "watchlist.yaml"
    wl_path.write_text("arxiv:\n  categories: [quant-ph]\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "watch",
            "--watchlist",
            str(wl_path),
            "refresh",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "arxiv:2408.13687v1" in result.output
    assert "Stim README" in result.output
