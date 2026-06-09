"""arXiv fetcher — parses Atom responses and produces chunks.

Network fetching uses the `arxiv` library (already used by the resolver).
This module's parser is decoupled so we can unit-test on fixtures.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text

_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    pdf_url: str


def parse_arxiv_atom(xml_text: str) -> ArxivMetadata:
    """Parse arXiv Atom response and extract metadata from the first entry.

    Args:
        xml_text: XML string from arXiv API Atom feed.

    Returns:
        ArxivMetadata with extracted arxiv_id, title, authors, year, abstract, pdf_url.

    Raises:
        ValueError: If the XML has no <entry> element or is malformed.
    """
    root = ET.fromstring(xml_text)
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        raise ValueError("arxiv atom: no <entry> element")

    # Extract arxiv_id from id element
    id_elem = entry.find(f"{_ATOM}id")
    id_text = ""
    if id_elem is not None and id_elem.text:
        id_text = id_elem.text.strip()
    m = re.search(r"abs/([^\s]+)", id_text)
    arxiv_id = m.group(1) if m else id_text

    # Extract title
    title_elem = entry.find(f"{_ATOM}title")
    title = ""
    if title_elem is not None and title_elem.text:
        title = title_elem.text.strip().replace("\n", " ")

    # Extract summary/abstract
    summary_elem = entry.find(f"{_ATOM}summary")
    summary = ""
    if summary_elem is not None and summary_elem.text:
        summary = summary_elem.text.strip()

    # Extract published year
    pub_elem = entry.find(f"{_ATOM}published")
    published = ""
    if pub_elem is not None and pub_elem.text:
        published = pub_elem.text[:4]
    year = int(published) if published.isdigit() else 0

    # Extract authors
    authors: list[str] = []
    for author_elem in entry.findall(f"{_ATOM}author"):
        name_elem = author_elem.find(f"{_ATOM}name")
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text)

    # Extract PDF URL
    pdf_url = ""
    for link_elem in entry.findall(f"{_ATOM}link"):
        if link_elem.get("type") == "application/pdf":
            pdf_url = link_elem.get("href") or ""
            break

    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        year=year,
        abstract=summary,
        pdf_url=pdf_url,
    )


def fetch_arxiv_by_id(arxiv_id: str) -> ArxivMetadata:
    """Trusted-id fetch (only used after the user explicitly accepts §7.4 warning)."""
    import arxiv

    [result] = list(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    # Re-use the Atom parser by emitting metadata directly
    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=result.title.replace("\n", " ").strip(),
        authors=[a.name for a in result.authors],
        year=result.published.year,
        abstract=result.summary,
        pdf_url=result.pdf_url or "",
    )


def paper_to_chunks(
    md: ArxivMetadata,
    *,
    body_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Convert arXiv paper metadata and text to a list of chunks.

    Args:
        md: ArxivMetadata with paper information.
        body_text: Full text to chunk (e.g., abstract or body).
        target_tokens: Target chunk size in tokens (~4 chars/token).
        overlap_tokens: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects with source_type='arxiv' and authority_tier='preprint'.
    """
    source_id = f"arxiv:{md.arxiv_id}"
    pieces = chunk_text(body_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    chunks: list[Chunk] = []
    for i, p in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}#{i}",
                source_id=source_id,
                chunk_index=i,
                text=p,
                title=md.title,
                source_type="arxiv",
                authority_tier="preprint",
                anchor=f"#chunk-{i}",
                metadata={"authors": md.authors, "year": md.year, "pdf_url": md.pdf_url},
            )
        )
    return chunks
