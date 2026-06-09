"""URL fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.fetchers.url import (
    URLMetadata,
    markdown_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "blog_post.md"


def test_markdown_to_chunks_uses_section_boundaries():
    md_text = FIXTURE.read_text(encoding="utf-8")
    md = URLMetadata(
        url="https://research.example.com/surface-code-milestone",
        title="Surface codes hit a milestone",
        publish_date="2026-01-15",
    )
    chunks = markdown_to_chunks(md, markdown_text=md_text, target_tokens=120, overlap_tokens=10)
    assert len(chunks) >= 2
    assert any("Willow" in c.text for c in chunks)
    assert chunks[0].source_type == "blog"
    assert chunks[0].authority_tier == "blog"


def test_url_source_id_is_hash_prefixed():
    md = URLMetadata(url="https://a.test/x", title="t", publish_date="")
    chunks = markdown_to_chunks(md, markdown_text="body", target_tokens=100, overlap_tokens=5)
    assert chunks[0].source_id.startswith("url:")
    assert len(chunks[0].source_id) == len("url:") + 16
