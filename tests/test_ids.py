"""Canonical source-ID generators per source type.

The IDs are deterministic strings used as the registry primary key and as the
prefix for chunk IDs. They must be stable across runs and across machines.
"""

import pytest

from chatqec.ingestion.ids import (
    canonical_wiki_id,
    github_id,
    pdf_id,
    url_id,
    youtube_id,
    arxiv_id,
)


def test_arxiv_id_includes_version():
    assert arxiv_id("2408.13687", "v1") == "arxiv:2408.13687v1"
    assert arxiv_id("quant-ph/9705052", "v2") == "arxiv:quant-ph/9705052v2"


def test_arxiv_id_defaults_to_v1_when_version_missing():
    assert arxiv_id("2408.13687") == "arxiv:2408.13687v1"


def test_github_id_uses_org_repo_and_short_sha():
    assert github_id("quantumlib", "Stim", "deadbeefcafef00d") == "github:quantumlib/Stim@deadbeef"


def test_youtube_id_strips_query_and_keeps_video_id():
    assert youtube_id("https://www.youtube.com/watch?v=abc123XYZ&t=42s") == "youtube:abc123XYZ"
    assert youtube_id("https://youtu.be/abc123XYZ") == "youtube:abc123XYZ"


def test_url_id_hashes_url():
    # url IDs use a short hex prefix of sha256 for compactness
    assert url_id("https://example.com/post").startswith("url:")
    assert len(url_id("https://example.com/post")) == len("url:") + 16


def test_pdf_id_uses_content_hash_prefix():
    assert pdf_id("ab12cd34" + "0" * 56).startswith("pdf:")
    assert pdf_id("ab12cd34" + "0" * 56) == "pdf:ab12cd34"


def test_canonical_wiki_id_uses_topic_slug():
    assert canonical_wiki_id("surface-code") == "canonical:surface-code"


def test_canonical_wiki_id_rejects_uppercase_and_spaces():
    with pytest.raises(ValueError):
        canonical_wiki_id("Surface Code")
    with pytest.raises(ValueError):
        canonical_wiki_id("surface_code")  # underscore disallowed (kebab-case only)
