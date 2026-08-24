"""ask()/ask_stream() offer wiki sections to the reranker alongside corpus chunks."""

from unittest.mock import MagicMock

from chatqec.api import ChatQEC
from chatqec.conversation.classifier import TopicVerdict
from chatqec.conversation.rewriter import RewriteResult
from chatqec.conversation.synthesizer import SynthesisResult
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk


class RecordingRetriever:
    """Captures what ask() hands the retriever, which is the whole point here."""

    def __init__(self):
        self.extra_seen = None

    def retrieve(self, *, queries, top_k, candidate_k, extra_chunks=()):
        self.extra_seen = list(extra_chunks)
        return RetrievalResult(chunks=[], matched_topics=[], debug={"top_rerank_score": 0.9})


def _wiki_chunk():
    return RetrievedChunk(
        chunk_id="qappswiki:concepts/qec/steane#Description",
        source_id="qappswiki:concepts/qec/steane",
        text="The [[7,1,3]] CSS code.",
        title="Steane code (QAppsWiki, CC-BY-SA Error Correction Zoo)",
        authority_tier="wiki_unverified",
        score=4.2,
        payload={},
    )


def _bot(retriever, wiki_searcher):
    """Same MagicMock doubles as tests/test_api_chatqec.py's _build_chatqec helper."""
    classifier = MagicMock()
    classifier.classify.return_value = TopicVerdict(on_topic=True, refusal_message="")
    rewriter = MagicMock()
    rewriter.rewrite.return_value = RewriteResult(rewritten="q", variants=[])
    canonical_index = MagicMock()
    canonical_index.pages = []
    synthesizer = MagicMock()
    synthesizer.mcp_adapter = None
    synthesizer.synthesize.return_value = SynthesisResult(
        text="Answer.", citations=[], confidence="medium"
    )
    synthesizer.synthesize_stream.return_value = iter(["Answer."])
    return ChatQEC(
        classifier=classifier,
        rewriter=rewriter,
        retriever=retriever,
        synthesizer=synthesizer,
        canonical_index=canonical_index,
        wiki_searcher=wiki_searcher,
    )


def test_wiki_searcher_output_reaches_the_retriever():
    retriever = RecordingRetriever()
    _bot(retriever, lambda q, k: [_wiki_chunk()]).ask("what is the steane code?")
    assert [c.source_id for c in retriever.extra_seen] == ["qappswiki:concepts/qec/steane"]


def test_no_wiki_searcher_means_no_extra_chunks():
    retriever = RecordingRetriever()
    _bot(retriever, None).ask("what is the steane code?")
    assert retriever.extra_seen == []


def test_a_raising_wiki_searcher_never_breaks_the_answer():
    def boom(q, k):
        raise TimeoutError("wiki is down")

    retriever = RecordingRetriever()
    _bot(retriever, boom).ask("what is the steane code?")  # must not raise
    assert retriever.extra_seen == []


def test_wiki_searcher_output_reaches_the_retriever_via_ask_stream():
    """Mirrors the ask() test above for the streaming call site (api.py's second
    ``retrieve()`` call, at ~line 265 pre-Task-8 — the brief flagged missing this
    one as the most likely way to get the task wrong)."""
    retriever = RecordingRetriever()
    bot = _bot(retriever, lambda q, k: [_wiki_chunk()])
    list(bot.ask_stream("what is the steane code?"))  # consume the generator
    assert [c.source_id for c in retriever.extra_seen] == ["qappswiki:concepts/qec/steane"]


class RecordingWiki:
    """Captures exactly which string the wiki searcher was handed."""

    def __init__(self):
        self.queries = []

    def __call__(self, query, k):
        self.queries.append(query)
        return [_wiki_chunk()]


def _bot_with_rewrite(retriever, wiki, rewritten, variants=()):
    bot = _bot(retriever, wiki)
    bot.rewriter.rewrite.return_value = RewriteResult(
        rewritten=rewritten, variants=list(variants), is_followup=True
    )
    return bot


def test_the_wiki_is_searched_with_the_rewritten_query_not_the_raw_question():
    """The wiki serves a lexical BM25 index, which is precisely the retriever that
    cannot recover an implicit topic. On a follow-up the raw question carries no
    topic terms at all. It is also what the reranker scores against (queries[0])."""
    wiki = RecordingWiki()
    bot = _bot_with_rewrite(RecordingRetriever(), wiki, rewritten="surface code threshold")
    bot.ask("what about its threshold?")
    assert wiki.queries == ["surface code threshold"]


def test_ask_stream_also_searches_the_wiki_with_the_rewritten_query():
    wiki = RecordingWiki()
    bot = _bot_with_rewrite(RecordingRetriever(), wiki, rewritten="surface code threshold")
    list(bot.ask_stream("what about its threshold?"))
    assert wiki.queries == ["surface code threshold"]


def test_the_wiki_query_is_the_same_string_the_reranker_scores_against():
    """queries[0] is both what retrieve() reranks against and what the wiki gets,
    so a wiki hit and a corpus hit are judged against the same question."""

    class QueryCapturingRetriever(RecordingRetriever):
        def __init__(self):
            super().__init__()
            self.queries_seen = None

        def retrieve(self, *, queries, top_k, candidate_k, extra_chunks=()):
            self.queries_seen = list(queries)
            return super().retrieve(
                queries=queries, top_k=top_k, candidate_k=candidate_k, extra_chunks=extra_chunks
            )

    retriever = QueryCapturingRetriever()
    wiki = RecordingWiki()
    bot = _bot_with_rewrite(retriever, wiki, rewritten="steane code distance", variants=["css"])
    bot.ask("and its distance?")
    assert wiki.queries == [retriever.queries_seen[0]] == ["steane code distance"]
