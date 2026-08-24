"""Wiki sections enter retrieval as ordinary chunks, and never break the answer."""

from chatqec.retrieval.scoring import TIER_WEIGHTS
from chatqec.retrieval.wiki_search import wiki_chunks

ROW = {
    "path": "concepts/qec/steane.md",
    "node_id": "concepts/qec/steane",
    "title": "Steane code",
    "heading": "Description",
    "text": "The [[7,1,3]] CSS code built from the Hamming code.",
    "score": 4.2,
    "provenance_status": "needs-verification",
    "sources": ["https://errorcorrectionzoo.org/c/steane"],
    "url": "https://github.com/QSCSoftwareEcosystem/QAppsWiki/blob/main/concepts/qec/steane.md",
}


class FakeClient:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if self._raises:
            raise self._raises
        return self._result


def test_maps_a_row_onto_a_retrieved_chunk():
    [chunk] = wiki_chunks(FakeClient({"rows": [ROW]}), "steane code", k=8)
    assert chunk.source_id == "qappswiki:concepts/qec/steane"
    assert chunk.chunk_id == "qappswiki:concepts/qec/steane#Description"
    assert chunk.title.startswith("Steane code")
    assert "Hamming code" in chunk.text
    assert chunk.score == 4.2
    assert chunk.payload["url"].endswith("concepts/qec/steane.md")
    assert chunk.payload["sources"] == ["https://errorcorrectionzoo.org/c/steane"]


def test_unverified_pages_are_tiered_below_blog_and_curated_ones_above():
    [unverified] = wiki_chunks(FakeClient({"rows": [ROW]}), "q")
    assert unverified.authority_tier == "wiki_unverified"
    [curated] = wiki_chunks(
        FakeClient({"rows": [{**ROW, "provenance_status": "source-backed"}]}), "q"
    )
    assert curated.authority_tier == "wiki_curated"
    assert TIER_WEIGHTS["wiki_unverified"] < TIER_WEIGHTS["blog"] < TIER_WEIGHTS["wiki_curated"]
    assert TIER_WEIGHTS["wiki_curated"] < TIER_WEIGHTS["peer_reviewed"]


def test_zoo_imported_pages_carry_their_attribution_in_the_citation_title():
    # citations.py renders only `title` and `url`, so attribution has to ride on the
    # title or the reader never sees the CC-BY-SA requirement the Zoo imposes.
    [chunk] = wiki_chunks(FakeClient({"rows": [ROW]}), "q")
    assert chunk.title == "Steane code (QAppsWiki, CC-BY-SA Error Correction Zoo)"
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"


def test_non_zoo_pages_are_only_marked_as_wiki():
    row = {**ROW, "sources": ["../OpenQEvo/README.md"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki)"
    assert chunk.payload["attribution"] == ""


def test_passes_query_and_k_through_to_the_tool():
    client = FakeClient({"rows": []})
    wiki_chunks(client, "surface code", k=3, domain="quantum-error-correction")
    assert client.calls == [
        ("search_pages", {"query": "surface code", "k": 3, "domain": "quantum-error-correction"})
    ]


def test_domain_omitted_when_none():
    client = FakeClient({"rows": []})
    wiki_chunks(client, "surface code", k=3)
    assert client.calls == [("search_pages", {"query": "surface code", "k": 3})]


def test_a_bare_list_payload_is_accepted_too():
    assert len(wiki_chunks(FakeClient([ROW]), "q")) == 1


def test_every_failure_degrades_to_empty():
    assert wiki_chunks(None, "q") == []
    assert wiki_chunks(FakeClient(raises=TimeoutError("slow")), "q") == []
    assert wiki_chunks(FakeClient(raises=KeyError("search_pages")), "q") == []
    assert wiki_chunks(FakeClient({"rows": "not a list"}), "q") == []
    assert wiki_chunks(FakeClient({"rows": [{"junk": 1}, ROW]}), "q")[0].title.startswith(
        "Steane code"
    )


def test_non_iterable_sources_are_skipped_safely():
    # sources is a scalar (int); should skip it and still return the valid row.
    row_with_scalar_sources = {**ROW, "sources": 1}
    chunks = wiki_chunks(FakeClient({"rows": [row_with_scalar_sources, ROW]}), "q")
    assert len(chunks) == 2
    # First chunk has scalar sources, which becomes empty list and no attribution
    assert chunks[0].title == "Steane code (QAppsWiki)"
    assert chunks[0].payload["sources"] == []
    assert chunks[0].payload["attribution"] == ""
    # Second chunk has valid Zoo sources
    assert "CC-BY-SA Error Correction Zoo" in chunks[1].title


def test_lookalike_domains_do_not_earn_zoo_attribution():
    # Domain that merely contains "errorcorrectionzoo.org" as a substring should not match
    row = {**ROW, "sources": ["https://not-errorcorrectionzoo.org.evil.com/page"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki)"
    assert chunk.payload["attribution"] == ""


def test_real_zoo_urls_earn_attribution():
    row = {**ROW, "sources": ["https://errorcorrectionzoo.org/c/steane"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert "CC-BY-SA Error Correction Zoo" in chunk.title
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"


def test_relative_path_sources_do_not_crash_or_attribute():
    row = {**ROW, "sources": ["../OpenQEvo/README.md", "raw/error-correction-zoo.md"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki)"
    assert chunk.payload["attribution"] == ""


# --------------------------------------------------------------------------- #
# attribution driven by the authoritative frontmatter field
# --------------------------------------------------------------------------- #


def test_imported_from_drives_attribution_without_looking_at_sources():
    """`imported_from` is page frontmatter, so it is the authoritative licence
    signal. A `sources` URL is prose an editor can move into the page body."""
    row = {**ROW, "imported_from": "error-correction-zoo", "sources": ["../local/notes.md"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki, CC-BY-SA Error Correction Zoo)"
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"
    assert chunk.payload["imported_from"] == "error-correction-zoo"


def test_sources_host_stays_a_fallback_for_rows_without_the_field():
    """Older wiki versions serve no `imported_from`; those rows must not silently
    lose the CC-BY-SA notice."""
    assert "imported_from" not in ROW
    [chunk] = wiki_chunks(FakeClient({"rows": [ROW]}), "q")
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"
    assert chunk.payload["imported_from"] == ""


def test_an_empty_imported_from_still_falls_back_to_sources():
    row = {**ROW, "imported_from": ""}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"


def test_qem_zoo_imports_get_no_attribution_pending_a_licence_decision():
    """concepts/qem pages carry imported_from: qem-zoo. Their terms are unknown, so
    inventing an attribution string would be worse than omitting one."""
    row = {**ROW, "imported_from": "qem-zoo", "sources": ["https://qemzoo.com/c/zne"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki)"
    assert chunk.payload["attribution"] == ""


def test_an_unrelated_imported_from_does_not_earn_zoo_attribution():
    row = {**ROW, "imported_from": "somewhere-else", "sources": ["../local/notes.md"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.payload["attribution"] == ""


# --------------------------------------------------------------------------- #
# a missing tool is a configuration fact, not an error
# --------------------------------------------------------------------------- #


def test_a_missing_search_pages_tool_logs_at_debug_without_a_traceback(caplog):
    """The documented default (CHATQEC_MCP_SERVER=chatqec-mcp-tools, no wiki server)
    raises KeyError on every question. A warning with exc_info=True there means a
    full traceback per query, twice a turn under Streamlit."""
    import logging

    with caplog.at_level(logging.DEBUG, logger="chatqec.retrieval.wiki_search"):
        assert wiki_chunks(FakeClient(raises=KeyError("search_pages")), "q") == []
    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    assert record.exc_info is None


def test_a_real_transport_failure_still_warns(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="chatqec.retrieval.wiki_search"):
        assert wiki_chunks(FakeClient(raises=ConnectionResetError("pipe died")), "q") == []
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None


def test_a_timing_out_client_degrades_to_the_corpus_alone():
    """StdioMCPClient now raises TimeoutError instead of blocking forever; the
    answer must proceed without the wiki rather than propagate or hang."""
    assert wiki_chunks(FakeClient(raises=TimeoutError("MCP call exceeded 5s")), "q") == []
