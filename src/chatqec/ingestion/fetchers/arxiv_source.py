r"""arXiv LaTeX-source fetcher.

arXiv exposes the original LaTeX source via ``Result.download_source()``. This
module downloads that tarball, extracts the ``.tex`` files, and (via
``chatqec.ingestion.latex``) assembles them into clean, chunk-ready text.

The tarball-extraction logic is unit-tested with in-memory archives; the
network download is a thin wrapper (exercised by integration/network tests).
"""

from __future__ import annotations

import gzip
import io
import tarfile

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.fetchers.arxiv import ArxivMetadata, fetch_arxiv_by_id, paper_to_chunks
from chatqec.ingestion.latex import source_files_to_text


def _looks_like_latex(text: str) -> bool:
    """Heuristic: real LaTeX source contains backslash commands."""
    return "\\documentclass" in text or "\\begin{document}" in text or "\\input" in text


def extract_tex_from_tar(blob: bytes) -> dict[str, str]:
    """Return ``{filename: text}`` for every ``.tex`` file in an arXiv source blob.

    Handles a gzipped tar (common), a bare gzipped single ``.tex`` (older
    submissions), and a raw uncompressed ``.tex``. Returns an empty dict when
    the blob is not LaTeX source at all (e.g. arXiv served a PDF), so the caller
    can fall back to PDF parsing instead of decoding binary into garbage.
    """
    # arXiv serves a bare PDF when no source is available.
    if blob[:4] == b"%PDF":
        return {}

    # Try tar (optionally gzip-compressed) first.
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            out: dict[str, str] = {}
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".tex"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                out[member.name] = f.read().decode("utf-8", errors="replace")
            return out
    except tarfile.ReadError:
        pass

    # A single gzipped .tex (no tar wrapper).
    try:
        text = gzip.decompress(blob).decode("utf-8", errors="replace")
        return {"main.tex": text} if _looks_like_latex(text) else {}
    except (OSError, gzip.BadGzipFile):
        pass

    # A raw, uncompressed .tex submission. Decode strictly so binary blobs
    # (e.g. unrecognised formats) fail closed rather than becoming garbage text.
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    return {"main.tex": text} if _looks_like_latex(text) else {}


def download_arxiv_source(arxiv_id: str) -> bytes:  # pragma: no cover - network
    """Download the raw LaTeX-source blob for an arXiv id.

    Fetches the e-print endpoint with ``requests`` (streamed, with a
    Content-Length check) rather than ``arxiv.Result.download_source``, whose
    ``urlretrieve`` truncates large sources (ContentTooShortError).
    """
    import requests

    # The e-print endpoint serves the original source tarball.
    bare_id = arxiv_id.split("v", maxsplit=1)[0] if "v" in arxiv_id else arxiv_id
    url = f"https://arxiv.org/e-print/{bare_id}"
    headers = {"User-Agent": "chatqec/0.1 (+https://errorcorrectionzoo.org)"}
    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()
    blob = bytes(resp.content)
    expected = resp.headers.get("Content-Length")
    if expected is not None and len(blob) < int(expected):
        raise OSError(f"arxiv source for {arxiv_id} truncated: got {len(blob)} of {expected} bytes")
    return blob


def fetch_arxiv_fulltext(
    arxiv_id: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> tuple[ArxivMetadata, list[Chunk]]:  # pragma: no cover - network
    """Fetch metadata + full LaTeX-source text for an arXiv id and chunk it.

    Raises ``SourceUnavailableError`` if no usable ``.tex`` source is found, so
    the caller can fall back to PDF parsing.
    """
    md = fetch_arxiv_by_id(arxiv_id)
    blob = download_arxiv_source(arxiv_id)
    files = extract_tex_from_tar(blob)
    if not files:
        raise SourceUnavailableError(arxiv_id)
    body = source_files_to_text(files)
    if not body.strip():
        raise SourceUnavailableError(arxiv_id)
    chunks = paper_to_chunks(
        md, body_text=body, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    )
    return md, chunks


def fetch_arxiv_pdf_fulltext(
    arxiv_id: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> tuple[ArxivMetadata, list[Chunk]]:  # pragma: no cover - network + subprocess
    """Fallback for PDF-only papers: download the PDF, extract with isolated Marker.

    Marker reconstructs equations to LaTeX and preserves code/tables, then the
    Markdown flows through the same math-aware chunker.
    """
    import tempfile
    from pathlib import Path

    import requests

    from chatqec.ingestion.fetchers.pdf_marker import run_marker_on_pdf

    md = fetch_arxiv_by_id(arxiv_id)
    bare_id = arxiv_id.split("v", maxsplit=1)[0] if "v" in arxiv_id else arxiv_id
    url = f"https://arxiv.org/pdf/{bare_id}"
    headers = {"User-Agent": "chatqec/0.1 (+https://errorcorrectionzoo.org)"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{bare_id.replace('/', '_')}.pdf"
        pdf_path.write_bytes(bytes(resp.content))
        markdown = run_marker_on_pdf(pdf_path)
    chunks = paper_to_chunks(
        md, body_text=markdown, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    )
    return md, chunks


class SourceUnavailableError(RuntimeError):
    """Raised when an arXiv paper has no usable LaTeX source (need PDF fallback)."""
