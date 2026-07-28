"""CLI tests for `chatqec ingest --github`.

No network access or GITHUB_TOKEN is required here: --github without
--commit-sha must fail validation before any network call is attempted.
"""

from click.testing import CliRunner

from chatqec.cli import main


def test_ingest_github_requires_commit_sha():
    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--github", "QuTone/LightStim"])
    assert result.exit_code != 0
    assert "--commit-sha" in result.output
