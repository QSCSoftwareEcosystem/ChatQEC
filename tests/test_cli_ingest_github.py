"""CLI tests for `chatqec ingest --github`.

No network access or GITHUB_TOKEN is required here: --github without
--commit-sha must fail validation before any network call is attempted.
"""

from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main


def test_ingest_github_requires_commit_sha(tmp_path: Path):
    # --chunks-dir/--registry point at tmp_path so this never touches the real,
    # git-tracked data/chunks or sources/registry.jsonl, matching the isolation
    # pattern used by every other test in tests/test_cli_ingest.py.
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--chunks-dir",
            str(tmp_path / "chunks"),
            "--registry",
            str(tmp_path / "registry.jsonl"),
            "--github",
            "QuTone/LightStim",
        ],
    )
    assert result.exit_code != 0
    assert "--commit-sha" in result.output
