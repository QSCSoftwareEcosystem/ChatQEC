"""Manual-upload PDF fetcher.

Extraction uses `unstructured.partition.pdf.partition_pdf`. The public function
`extract_pdf_text` returns a single string; the parser/chunker is decoupled so
it can be tested without an actual PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.hashing import sha256_file
from chatqec.ingestion.ids import pdf_id

# Matches Markdown image references, e.g. ``![](data/figures/pdf_xxx/f1.png)``.
_CHUNK_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class PDFMetadata:
    content_sha256: str
    title: str
    author: str
    year: int
    source_path: str


def extract_pdf_text(path: Path) -> str:
    """Thin wrapper around `unstructured`. Not unit-tested — see network tests."""
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path))
    return "\n".join(str(el) for el in elements)


def build_pdf_metadata(
    path: Path,
    *,
    title: str,
    author: str,
    year: int,
) -> PDFMetadata:
    return PDFMetadata(
        content_sha256=sha256_file(path),
        title=title,
        author=author,
        year=year,
        source_path=str(path),
    )


def text_to_chunks(
    md: PDFMetadata,
    *,
    body_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    tier: str = "preprint",
    source_id_suffix: str = "",
) -> list[Chunk]:
    """Chunk extracted PDF text.

    ``tier`` sets the authority tier (books use ``peer_reviewed``).
    ``source_id_suffix`` disambiguates multiple chapters extracted from the same
    PDF file (identical content hash) — e.g. ``":pp425-499"`` — so per-chapter
    ingests don't collide on a single ``pdf:<sha>`` id.
    """
    source_id = pdf_id(md.content_sha256) + source_id_suffix
    pieces = chunk_text(body_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=p,
            title=md.title,
            source_type="pdf",
            authority_tier=tier,
            anchor=f"#chunk-{i}",
            metadata={
                "author": md.author,
                "year": md.year,
                "source_path": md.source_path,
                "figures": _CHUNK_IMG_RE.findall(p),
            },
        )
        for i, p in enumerate(pieces)
    ]
