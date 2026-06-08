"""Section-based chunker for canonical pages.

Canonical content is author-controlled and well-structured; we chunk on '##'
boundaries and emit one chunk per section.
"""

from chatqec.canonical.chunker import chunk_canonical_body


SAMPLE = """# Surface Code

Intro paragraph mentioning the toric code.

## Stabilizers

X-plaquettes and Z-stars.

## Decoding

MWPM on the syndrome graph.

### Subdecoders

Union-Find is faster.
"""


def test_chunker_emits_one_chunk_per_h2():
    chunks = chunk_canonical_body(SAMPLE)
    assert [c.section_title for c in chunks] == ["Preamble", "Stabilizers", "Decoding"]


def test_chunker_includes_h3_in_parent_h2():
    chunks = chunk_canonical_body(SAMPLE)
    decoding = next(c for c in chunks if c.section_title == "Decoding")
    assert "Union-Find" in decoding.text
    assert "### Subdecoders" in decoding.text


def test_chunker_preamble_captures_text_before_first_h2():
    chunks = chunk_canonical_body(SAMPLE)
    preamble = chunks[0]
    assert preamble.section_title == "Preamble"
    assert "toric code" in preamble.text


def test_chunker_anchor_is_kebab_of_section_title():
    chunks = chunk_canonical_body(SAMPLE)
    assert chunks[1].anchor == "stabilizers"
    assert chunks[2].anchor == "decoding"
    assert chunks[0].anchor == ""  # preamble has no heading anchor


def test_chunker_yields_no_chunks_for_empty_body():
    assert list(chunk_canonical_body("")) == []


def test_chunk_dataclass_is_frozen():
    chunks = chunk_canonical_body(SAMPLE)
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunks[0].text = "x"  # type: ignore[misc]
