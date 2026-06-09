"""System prompt + retrieved-context assembly."""

from __future__ import annotations

from chatqec.retrieval.hybrid import RetrievedChunk

SYNTHESIZER_SYSTEM_PROMPT = """You are ChatQEC, a research assistant for Quantum Error Correction.

You answer based ONLY on the retrieved context below. If the context does not
contain enough evidence, say so explicitly — never invent facts.

Citation rules:
- Cite every non-trivial claim using inline footnote markers like [^1], [^2].
- Footnote numbers refer to the numbered context entries below.
- At the end of your answer, include a "Sources:" line listing the cited footnote numbers.

Style:
- Be precise and technical; the user is a researcher.
- Use LaTeX for math: $..$, $$..$$, \\begin{equation}..\\end{equation}.
- Prefer canonical wiki entries (marked CANONICAL) for terminology and definitions.
- For questions beyond the retrieved context, refuse politely: "I don't have evidence in the knowledge base for this."

QEC focus only — refuse off-topic queries politely.
"""


def assemble_context_block(
    retrieved: list[RetrievedChunk],
    *,
    canonical_chunks: list[RetrievedChunk],
) -> str:
    if not retrieved and not canonical_chunks:
        return "Retrieved context: (none) — no relevant chunks matched the query\n"

    lines = ["Retrieved context:"]
    n = 1
    for c in canonical_chunks:
        lines.append(f"[^{n}] [CANONICAL · {c.title} · {c.source_id}] {c.text}")
        n += 1
    for c in retrieved:
        lines.append(f"[^{n}] [{c.authority_tier} · {c.title} · {c.source_id}] {c.text}")
        n += 1
    return "\n\n".join(lines)
