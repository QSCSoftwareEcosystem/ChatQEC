"""CLI tests for `chatqec ingest --github`.

No network access or GITHUB_TOKEN is required here: `ingest()` validates
--commit-sha presence/format, GITHUB_TOKEN presence, and --github's org/repo
shape *before* any llama_index import or network call is attempted, so all of
these are exercisable offline.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from chatqec.cli import main

_VALID_SHA = "47dd74976780192ae901d91958863cced1165354"


def _invoke(tmp_path: Path, args: list[str]):
    runner = CliRunner()
    return runner.invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(tmp_path / "chunks"),
            "--registry",
            str(tmp_path / "registry.jsonl"),
            *args,
        ],
    )


def test_ingest_github_requires_commit_sha(tmp_path: Path):
    # --chunks-dir/--registry point at tmp_path so this never touches the real,
    # git-tracked data/chunks or sources/registry.jsonl, matching the isolation
    # pattern used by every other test in tests/test_cli_ingest.py.
    result = _invoke(tmp_path, ["--github", "QuTone/LightStim"])
    assert result.exit_code != 0
    assert "--commit-sha" in result.output


def test_ingest_github_requires_github_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = _invoke(
        tmp_path,
        ["--github", "QuTone/LightStim", "--commit-sha", _VALID_SHA],
    )
    assert result.exit_code != 0
    assert "GITHUB_TOKEN" in result.output


def test_ingest_github_rejects_non_hex_commit_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A branch name (not a pinned commit sha) must be rejected before any
    # network call, regardless of whether GITHUB_TOKEN is set.
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    result = _invoke(
        tmp_path,
        ["--github", "QuTone/LightStim", "--commit-sha", "main"],
    )
    assert result.exit_code != 0
    assert "--commit-sha" in result.output


def test_ingest_github_rejects_malformed_org_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    result = _invoke(
        tmp_path,
        ["--github", "LightStim", "--commit-sha", _VALID_SHA],
    )
    assert result.exit_code != 0
    assert "--github" in result.output
