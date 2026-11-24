"""Literature code parameters (BB5, tile, directional tile, barbell): copied from the papers."""

import json
import os
import re
from pathlib import Path

import pytest

from chatqec_mcp_tools.tools._paper_codes import FAMILIES, PaperCode, paper_codes

_CHATQEC = Path(os.environ.get("CHATQEC_REPO", Path.home() / "chatQEC"))
_CHUNKS = _CHATQEC / "data" / "chunks"
_REGISTRY = _CHATQEC / "sources" / "registry.jsonl"


def _all_rows() -> list[PaperCode]:
    return [row for family in FAMILIES for row in paper_codes(family)]


def test_the_four_families_are_offered():
    assert set(FAMILIES) == {"bb5", "tile", "directional_tile", "barbell"}


def test_every_row_has_a_source_and_a_locator():
    for row in _all_rows():
        assert re.fullmatch(r"arXiv:\d{4}\.\d{4,5}", row.source), row
        assert row.locator.strip(), row


def test_parameters_are_sane():
    for row in _all_rows():
        assert 0 < row.k < row.n and 0 < row.d <= row.n, row


def test_rows_are_unique_per_source_and_locator():
    keys = [(r.family, r.n, r.k, r.d, r.source, r.locator) for r in _all_rows()]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "family,triple,source",
    [
        ("bb5", (30, 4, 5), "arXiv:2503.22071"),  # Ye and Delfosse, check weight 5
        ("bb5", (30, 4, 5), "arXiv:2606.06455"),  # hardware demo, [[6d,4,d]] with d = 5
        ("bb5", (48, 4, 7), "arXiv:2503.22071"),
        ("tile", (288, 8, 12), "arXiv:2504.09171"),
        ("tile", (512, 18, 19), "arXiv:2504.09171"),
        ("directional_tile", (323, 14, 15), "arXiv:2606.19482"),
        ("barbell", (288, 16, 12), "arXiv:2606.06062"),
    ],
)
def test_known_rows_are_present_with_their_source(family, triple, source):
    rows = [(r.n, r.k, r.d, r.source) for r in paper_codes(family)]
    assert (*triple, source) in rows


def test_unknown_family_is_an_error():
    with pytest.raises(KeyError):
        paper_codes("surface_code_rotated")


@pytest.mark.skipif(not _CHUNKS.is_dir(), reason="chatQEC chunk files not available")
def test_every_row_appears_in_the_ingested_text_of_its_source():
    """Guards against a mistyped value: the triple must be in the paper we ingested."""
    cache: dict[str, str] = {}
    for row in _all_rows():
        path = _CHUNKS / f"arxiv_{row.source.split(':')[1]}.jsonl"
        if not path.exists():
            pytest.skip(f"{path.name} not ingested")
        text = cache.setdefault(
            row.source, " ".join(json.loads(line)["text"] for line in path.read_text().splitlines())
        )
        pattern = rf"(?<!\d){row.n}\s*,\s*{row.k}\s*,\s*{row.d}(?!\d)"
        assert re.search(pattern, text), f"{row} not found in {path.name}"


@pytest.mark.skipif(not _REGISTRY.exists(), reason="chatQEC source registry not available")
def test_every_source_is_in_the_chatqec_registry():
    ids = {json.loads(line)["id"] for line in _REGISTRY.read_text().splitlines() if line.strip()}
    for source in {r.source for r in _all_rows()}:
        assert source.lower() in ids, f"{source} is not an ingested source"
