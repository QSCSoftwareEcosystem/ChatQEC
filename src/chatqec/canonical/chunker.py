"""Section-based chunker for canonical wiki pages.

Splits markdown body into chunks at each '##' heading. Subheadings (###, ####)
stay within their parent ## section. Text before the first ## (if any) becomes
a "Preamble" chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalChunk:
    section_title: str
    text: str
    anchor: str  # kebab-case version of section_title, "" for preamble


_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)


def _slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s


def chunk_canonical_body(body: str) -> list[CanonicalChunk]:
    if not body.strip():
        return []

    # Locate all H2 headings. Each H2 starts a section that runs until the next H2.
    matches = list(_H2_RE.finditer(body))
    chunks: list[CanonicalChunk] = []

    if not matches:
        # Whole body is preamble
        return [CanonicalChunk(section_title="Preamble", text=body.strip(), anchor="")]

    # Preamble = everything before first H2
    first = matches[0]
    preamble = body[: first.start()].strip()
    if preamble:
        chunks.append(CanonicalChunk(section_title="Preamble", text=preamble, anchor=""))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        chunks.append(
            CanonicalChunk(section_title=title, text=section_text, anchor=_slugify(title))
        )
    return chunks
