"""One card per source: several chunks of one paper must not become several cards."""

from chatqec.api import Citation
from chatqec_app.components.citation_card import group_by_source


def _c(source: str, chunk: int, marker: int) -> Citation:
    return Citation(
        source_id=source,
        chunk_id=f"{source}#{chunk}",
        title=f"T {source}",
        url=f"https://x/{source}",
        marker=marker,
    )


def test_chunks_of_one_source_share_one_card_in_first_cited_order():
    cites = [_c("arxiv:a", 0, 1), _c("arxiv:b", 0, 2), _c("arxiv:a", 3, 3), _c("arxiv:a", 5, 4)]
    groups = group_by_source(cites)
    assert [g.source_id for g in groups] == ["arxiv:a", "arxiv:b"]
    assert groups[0].markers == [1, 3, 4]
    assert groups[0].chunk_ids == ["arxiv:a#0", "arxiv:a#3", "arxiv:a#5"]
    assert groups[0].label == "[^1][^3][^4] [arxiv:a] T arxiv:a"
