"""Deterministic source-ID generators.

Each registry record's primary key is one of:
    arxiv:<id>v<version>
    github:<org>/<repo>@<short_sha>
    youtube:<video_id>
    url:<sha256[:16]>
    pdf:<content_sha256[:8]>
    canonical:<topic-slug>
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from chatqec.ingestion.hashing import sha256_text

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PDF_ID_HASH_PREFIX_LEN = 8
_URL_ID_HASH_PREFIX_LEN = 16
_GITHUB_SHA_TRUNCATE_LEN = 8


def arxiv_id(paper_id: str, version: str = "v1") -> str:
    """Generate canonical arXiv source ID.

    Args:
        paper_id: The arXiv paper identifier (e.g., "2408.13687" or "quant-ph/9705052").
        version: The arXiv version string; defaults to "v1". If not prefixed with 'v',
                 'v' is prepended.

    Returns:
        Canonical ID in the form "arxiv:<paper_id>v<version>".
    """
    if not version.startswith("v"):
        version = f"v{version}"
    return f"arxiv:{paper_id}{version}"


def github_id(org: str, repo: str, commit_sha: str) -> str:
    """Generate canonical GitHub source ID.

    Args:
        org: GitHub organization name.
        repo: Repository name.
        commit_sha: Full or partial commit SHA; only the first 8 characters are used.

    Returns:
        Canonical ID in the form "github:<org>/<repo>@<short_sha>".
    """
    return f"github:{org}/{repo}@{commit_sha[:_GITHUB_SHA_TRUNCATE_LEN]}"


def youtube_id(url: str) -> str:
    """Generate canonical YouTube source ID.

    Args:
        url: A YouTube URL (e.g., "https://www.youtube.com/watch?v=abc123XYZ" or
             "https://youtu.be/abc123XYZ").

    Returns:
        Canonical ID in the form "youtube:<video_id>".

    Raises:
        ValueError: If the video ID cannot be extracted from the URL.
    """
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        vid = parsed.path.lstrip("/")
    else:
        vid = parse_qs(parsed.query).get("v", [""])[0]
    if not vid:
        raise ValueError(f"could not extract YouTube video id from {url!r}")
    return f"youtube:{vid}"


def url_id(url: str) -> str:
    """Generate canonical URL source ID.

    Args:
        url: A full URL string.

    Returns:
        Canonical ID in the form "url:<sha256_hex[:16]>", where the hash provides
        a compact unique identifier for the URL.
    """
    return f"url:{sha256_text(url)[:_URL_ID_HASH_PREFIX_LEN]}"


def pdf_id(content_sha256_hex: str) -> str:
    """Generate canonical PDF source ID.

    Args:
        content_sha256_hex: The full SHA-256 hex digest of the PDF content (64 characters).

    Returns:
        Canonical ID in the form "pdf:<sha256_hex[:8]>".

    Raises:
        ValueError: If the provided hash is shorter than 8 characters.
    """
    if len(content_sha256_hex) < _PDF_ID_HASH_PREFIX_LEN:
        raise ValueError("content_sha256 must be a full hex digest")
    return f"pdf:{content_sha256_hex[:_PDF_ID_HASH_PREFIX_LEN]}"


def canonical_wiki_id(topic_slug: str) -> str:
    """Generate canonical wiki topic ID.

    Args:
        topic_slug: A kebab-case identifier (lowercase alphanumeric + hyphens only).

    Returns:
        Canonical ID in the form "canonical:<topic_slug>".

    Raises:
        ValueError: If the slug does not match the kebab-case pattern
                    (a-z, 0-9, hyphen only; no uppercase, spaces, or underscores).
    """
    if not _SLUG_RE.match(topic_slug):
        raise ValueError(
            f"canonical wiki topic_slug must be kebab-case (a-z, 0-9, hyphen): {topic_slug!r}"
        )
    return f"canonical:{topic_slug}"
