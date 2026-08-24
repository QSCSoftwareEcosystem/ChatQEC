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
