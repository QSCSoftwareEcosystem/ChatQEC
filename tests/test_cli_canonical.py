"""CLI tests for `chatqec canonical` subcommands."""

from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main


def _write(path: Path, slug: str, see_also: list[str] | None = None) -> None:
    see_also = see_also or []
    path.write_text(
        f"""---
topic_slug: {slug}
title: "{slug}"
aliases: []
see_also: {see_also!r}
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["m"]
---
# {slug}

body
""",
        encoding="utf-8",
    )


def test_canonical_list_prints_slugs(tmp_path: Path):
    _write(tmp_path / "a.md", "a")
    _write(tmp_path / "b.md", "b")
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "list"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" in result.output


def test_canonical_validate_passes_clean_dir(tmp_path: Path):
    _write(tmp_path / "a.md", "a", see_also=["b"])
    _write(tmp_path / "b.md", "b")
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "validate"])
    assert result.exit_code == 0
    assert "ok" in result.output.lower() or "2 pages" in result.output.lower()


def test_canonical_validate_reports_broken_cross_ref(tmp_path: Path):
    _write(tmp_path / "a.md", "a", see_also=["nonexistent"])
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "validate"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output.lower()


def test_canonical_embed_stage_only_writes_chunks_and_registry(tmp_path: Path):
    canon = tmp_path / "canonical"
    canon.mkdir()
    _write(canon / "a.md", "a")
    _write(canon / "b.md", "b")
    chunks_dir = tmp_path / "chunks"
    registry = tmp_path / "registry.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "canonical",
            "--dir",
            str(canon),
            "embed",
            "--stage-only",
            "--chunks-dir",
            str(chunks_dir),
            "--registry",
            str(registry),
        ],
    )
    assert result.exit_code == 0
    assert "staged 2 canonical pages" in result.output
    assert (chunks_dir / "canonical_a.jsonl").exists()
    assert (chunks_dir / "canonical_b.jsonl").exists()
    assert '"type":"canonical_wiki"' in registry.read_text()
