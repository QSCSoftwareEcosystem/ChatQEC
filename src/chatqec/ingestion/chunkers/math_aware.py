r"""Math-aware text chunker.

Splits text into ~target_tokens-sized chunks at sentence boundaries while
never splitting inside a protected LaTeX region:

    $...$        inline math
    $$...$$      display math
    \begin{X}...\end{X}   any environment

Protection is done by first masking protected spans, then chunking the masked
text by sentence, then unmasking before yielding.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Approximate token count from character count (4 chars/token rule of thumb).
_CHARS_PER_TOKEN = 4


_MASK_TOKEN = "☃PROT{}☃"  # snowman as unlikely-in-source delimiter


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    # Sort patterns from greediest to least greedy to avoid nested mis-matches.
    patterns = [
        (re.compile(r"\\begin\{(\w+)\}.*?\\end\{\1\}", re.DOTALL), 0),
        (re.compile(r"\$\$.*?\$\$", re.DOTALL), 0),
        (re.compile(r"\$[^$\n]+\$"), 0),  # inline math doesn't cross newlines
    ]
    occupied: list[tuple[int, int]] = []  # already-claimed [start,end) intervals
    for pat, _ in patterns:
        for m in pat.finditer(text):
            s, e = m.span()
            # skip if overlaps an earlier (broader) span
            if any(not (e <= os or s >= oe) for (os, oe) in occupied):
                continue
            occupied.append((s, e))
            spans.append((s, e))
    return sorted(spans)


def _mask_protected(text: str) -> tuple[str, list[str]]:
    """Replace protected spans with sentinel tokens; return (masked_text, originals)."""
    spans = _find_protected_spans(text)
    originals: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for s, e in spans:
        pieces.append(text[cursor:s])
        sentinel = _MASK_TOKEN.format(len(originals))
        originals.append(text[s:e])
        pieces.append(sentinel)
        cursor = e
    pieces.append(text[cursor:])
    return "".join(pieces), originals


def _unmask(text: str, originals: list[str]) -> str:
    for i, orig in enumerate(originals):
        text = text.replace(_MASK_TOKEN.format(i), orig)
    return text


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\\(])")


def _split_sentences(text: str) -> Iterable[str]:
    parts = _SENT_RE.split(text)
    for p in parts:
        if p:
            yield p


def _sub_split(segment: str, target_chars: int) -> Iterable[str]:
    """Split an oversized segment on whitespace boundaries into target_chars pieces.

    Used as a fallback when a 'sentence' (no sentence-ending punctuation) is
    longer than target_chars — e.g. a stream of tokens without any periods.
    """
    if len(segment) <= target_chars:
        yield segment
        return
    words = segment.split(" ")
    buf: list[str] = []
    buf_len = 0
    for word in words:
        if buf_len + len(word) + 1 > target_chars and buf:
            yield " ".join(buf)
            buf = []
            buf_len = 0
        buf.append(word)
        buf_len += len(word) + 1
    if buf:
        yield " ".join(buf)


def chunk_text(text: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into math-protected chunks of ~target_tokens characters of size."""
    if not text:
        return []
    target_chars = target_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    masked, originals = _mask_protected(text)
    if len(masked) <= target_chars:
        return [_unmask(masked, originals)]

    # Sentence-split, then sub-split any oversized segment by whitespace.
    raw_sentences = list(_split_sentences(masked))
    sentences: list[str] = []
    for sent in raw_sentences:
        sentences.extend(_sub_split(sent, target_chars))

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if buf_len + len(sent) + 1 > target_chars and buf:
            chunks.append(" ".join(buf))
            # build overlap tail
            tail: list[str] = []
            tlen = 0
            for s in reversed(buf):
                if tlen + len(s) > overlap_chars:
                    break
                tail.insert(0, s)
                tlen += len(s)
            buf = list(tail)
            buf_len = sum(len(s) for s in buf)
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        chunks.append(" ".join(buf))

    return [_unmask(c, originals) for c in chunks]
