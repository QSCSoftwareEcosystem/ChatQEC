"""CLI tests for `chatqec ingest --github`.

No network access or GITHUB_PERSONAL_ACCESS_TOKEN is required here: `ingest()` validates
--commit-sha presence/format, GITHUB_PERSONAL_ACCESS_TOKEN presence, and --github's org/repo
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
    # `main()`'s group callback calls `load_dotenv()` on every invocation, which
    # would silently resurrect a real GITHUB_PERSONAL_ACCESS_TOKEN from a
    # developer's local .env after delenv below — neutralize it so this test
    # verifies the "no token" branch regardless of the local .env's contents.
    monkeypatch.setattr("chatqec.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    result = _invoke(
        tmp_path,
        ["--github", "QuTone/LightStim", "--commit-sha", _VALID_SHA],
    )
    assert result.exit_code != 0
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in result.output


def test_ingest_github_rejects_non_hex_commit_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A branch name (not a pinned commit sha) must be rejected before any
    # network call, regardless of whether GITHUB_PERSONAL_ACCESS_TOKEN is set.
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "dummy")
    result = _invoke(
        tmp_path,
        ["--github", "QuTone/LightStim", "--commit-sha", "main"],
    )
    assert result.exit_code != 0
    assert "--commit-sha" in result.output


def test_ingest_github_rejects_malformed_org_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "dummy")
    result = _invoke(
        tmp_path,
        ["--github", "LightStim", "--commit-sha", _VALID_SHA],
    )
    assert result.exit_code != 0
    assert "--github" in result.output
