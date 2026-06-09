"""GitHub fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.fetchers.github import (
    GitHubMetadata,
    files_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "github_readme.md"


def test_files_to_chunks_emits_chunks_per_file():
    readme_text = FIXTURE.read_text(encoding="utf-8")
    md = GitHubMetadata(
        org="quantumlib",
        repo="Stim",
        commit_sha="deadbeefcafef00d",
        license="Apache-2.0",
    )
    chunks = files_to_chunks(
        md,
        files={"README.md": readme_text, "doc/intro.md": "# Intro\n\nShort intro."},
        target_tokens=100,
        overlap_tokens=10,
    )
    sources = {c.source_id for c in chunks}
    assert sources == {"github:quantumlib/Stim@deadbeef"}
    titles = {c.metadata.get("path") for c in chunks}
    assert "README.md" in titles
    assert "doc/intro.md" in titles


def test_files_to_chunks_skips_empty_files():
    md = GitHubMetadata(org="o", repo="r", commit_sha="abc12345", license="MIT")
    chunks = files_to_chunks(
        md,
        files={"empty.md": "", "real.md": "content"},
        target_tokens=100,
        overlap_tokens=10,
    )
    assert all(c.metadata.get("path") != "empty.md" for c in chunks)
    assert any(c.metadata.get("path") == "real.md" for c in chunks)


def test_files_to_chunks_assigns_blog_authority_tier_by_default():
    md = GitHubMetadata(org="o", repo="r", commit_sha="abc12345", license="MIT")
    chunks = files_to_chunks(md, files={"r.md": "x"}, target_tokens=100, overlap_tokens=10)
    assert chunks[0].source_type == "github"
    assert chunks[0].authority_tier == "blog"  # neither canonical nor peer-reviewed
