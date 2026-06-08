"""CLI tests for `chatqec sources` subcommands."""

from datetime import datetime, UTC
from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main
from chatqec.ingestion.registry import Registry, SourceRecord, SourceStatus, SourceType


def _seed(path: Path) -> None:
    reg = Registry(path)
    reg.append(
        SourceRecord(
            id="arxiv:2408.13687v1",
            type=SourceType.ARXIV,
            url="https://arxiv.org/abs/2408.13687",
            title="Quantum error correction below the surface code threshold",
            authors=["Rajeev Acharya"],
            year=2024,
            authority_tier="peer_reviewed",
            ingested_at=datetime(2026, 5, 29, tzinfo=UTC),
            last_checked_at=datetime(2026, 5, 29, tzinfo=UTC),
            status=SourceStatus.ACTIVE,
            verified=True,
        )
    )
    reg.append(
        SourceRecord(
            id="canonical:surface-code",
            type=SourceType.CANONICAL_WIKI,
            url="knowledge/canonical/surface-code.md",
            title="Surface Code",
            authority_tier="canonical",
            ingested_at=datetime(2026, 5, 29, tzinfo=UTC),
            last_checked_at=datetime(2026, 5, 29, tzinfo=UTC),
            status=SourceStatus.ACTIVE,
            verified=True,
        )
    )


def test_sources_list_prints_records(tmp_path: Path):
    _seed(tmp_path / "registry.jsonl")
    runner = CliRunner()
    result = runner.invoke(
        main, ["sources", "--registry", str(tmp_path / "registry.jsonl"), "list"]
    )
    assert result.exit_code == 0, result.output
    assert "arxiv:2408.13687v1" in result.output
    assert "canonical:surface-code" in result.output
    # Verify both records are shown in the table
    assert "2 records" in result.output


def test_sources_list_filters_by_status(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    _seed(path)
    reg = Registry(path)
    reg.update_status("arxiv:2408.13687v1", SourceStatus.RETRACTED)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sources",
            "--registry",
            str(path),
            "list",
            "--status",
            "active",
        ],
    )
    assert result.exit_code == 0
    assert "arxiv:2408.13687v1" not in result.output
    assert "canonical:surface-code" in result.output


def test_sources_show_prints_full_record(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    _seed(path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sources",
            "--registry",
            str(path),
            "show",
            "arxiv:2408.13687v1",
        ],
    )
    assert result.exit_code == 0
    assert "Rajeev Acharya" in result.output
    assert "peer_reviewed" in result.output


def test_sources_show_missing_id_exits_nonzero(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sources",
            "--registry",
            str(tmp_path / "registry.jsonl"),
            "show",
            "arxiv:nope",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
