"""Chunk = the unit the embedder consumes. M4 writes chunks to disk; M5 embeds them."""

import json
from pathlib import Path


from chatqec.ingestion.chunk import Chunk, ChunkSink


def test_chunk_required_fields():
    c = Chunk(
        chunk_id="arxiv:2408.13687v1#0",
        source_id="arxiv:2408.13687v1",
        chunk_index=0,
        text="The surface code is a topological code.",
        title="Quantum error correction below the surface code threshold",
        source_type="arxiv",
        authority_tier="peer_reviewed",
        anchor="#section-1",
    )
    assert c.chunk_id == "arxiv:2408.13687v1#0"
    assert c.metadata == {}


def test_chunk_round_trips_through_json():
    c = Chunk(
        chunk_id="x#0",
        source_id="x",
        chunk_index=0,
        text="hi",
        title="T",
        source_type="blog",
        authority_tier="blog",
        anchor="",
        metadata={"author": "alice"},
    )
    blob = c.model_dump_json()
    decoded = Chunk.model_validate_json(blob)
    assert decoded == c


def test_chunk_sink_writes_jsonl(tmp_path: Path):
    sink = ChunkSink(tmp_path)
    c1 = Chunk(
        chunk_id="x#0",
        source_id="x",
        chunk_index=0,
        text="a",
        title="T",
        source_type="blog",
        authority_tier="blog",
        anchor="",
    )
    c2 = Chunk(
        chunk_id="x#1",
        source_id="x",
        chunk_index=1,
        text="b",
        title="T",
        source_type="blog",
        authority_tier="blog",
        anchor="",
    )
    sink.write_many("x", [c1, c2])

    expected = tmp_path / "x.jsonl"
    assert expected.exists()
    lines = expected.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["text"] == "a"
    assert parsed[1]["chunk_index"] == 1


def test_chunk_sink_replaces_existing_file(tmp_path: Path):
    sink = ChunkSink(tmp_path)
    sink.write_many(
        "x",
        [
            Chunk(
                chunk_id="x#0",
                source_id="x",
                chunk_index=0,
                text="old",
                title="T",
                source_type="blog",
                authority_tier="blog",
                anchor="",
            ),
        ],
    )
    sink.write_many(
        "x",
        [
            Chunk(
                chunk_id="x#0",
                source_id="x",
                chunk_index=0,
                text="new",
                title="T",
                source_type="blog",
                authority_tier="blog",
                anchor="",
            ),
        ],
    )
    lines = (tmp_path / "x.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert "new" in lines[0]
