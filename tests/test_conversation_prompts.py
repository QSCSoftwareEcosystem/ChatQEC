"""System prompt + context-assembler tests."""

from chatqec.conversation.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    assemble_context_block,
)
from chatqec.retrieval.hybrid import RetrievedChunk


def _chunk(i: int, text: str = "body", tier: str = "preprint") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"x#{i}",
        source_id="x",
        text=text,
        title="T",
        authority_tier=tier,
        score=0.5,
        payload={},
    )


def test_system_prompt_mentions_citation_format():
    assert "[^" in SYNTHESIZER_SYSTEM_PROMPT
    assert "quantum error correction" in SYNTHESIZER_SYSTEM_PROMPT.lower()


def test_assemble_context_block_numbers_chunks_from_1():
    chunks = [_chunk(0, "chunk A"), _chunk(1, "chunk B")]
    block = assemble_context_block(chunks, canonical_chunks=[])
    assert "[^1]" in block
    assert "[^2]" in block
    assert "chunk A" in block
    assert "chunk B" in block


def test_canonical_chunks_appear_first_marked_as_canonical():
    retrieved = [_chunk(0, "regular chunk")]
    canonical = [_chunk(99, "canonical surface code intro", tier="canonical")]
    block = assemble_context_block(retrieved, canonical_chunks=canonical)
    assert block.index("canonical surface code intro") < block.index("regular chunk")
    assert "canonical" in block.lower()


def test_assemble_with_no_chunks_returns_explicit_empty_marker():
    block = assemble_context_block([], canonical_chunks=[])
    assert "no relevant" in block.lower() or "(none)" in block.lower()


def test_wiki_links_in_context_become_plain_text():
    """Canonical pages use Obsidian links. The model copied them into answers verbatim
    ("needs specialized [[color-code-decoders]]"), so they are flattened before the model
    sees the context; the canonical files themselves keep their links."""
    canon = _chunk(
        0, "needs [[color-code-decoders]] and [[color-code|color codes]] for [[css-codes]]"
    )
    block = assemble_context_block([], canonical_chunks=[canon])
    assert "[[" not in block
    assert "needs color code decoders and color codes for css codes" in block


def test_code_parameters_in_double_brackets_are_untouched():
    """[[n,k,d]] is QEC notation, not a link."""
    chunk = _chunk(0, "the [[72,12,6]] and [[ 144, 12, 12 ]] BB codes, and [[n,k,d]] in general")
    block = assemble_context_block([chunk], canonical_chunks=[])
    assert "[[72,12,6]]" in block and "[[ 144, 12, 12 ]]" in block and "[[n,k,d]]" in block
