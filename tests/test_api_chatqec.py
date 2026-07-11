"""Public ChatQEC class — end-to-end pipeline tests with all components stubbed."""

from unittest.mock import MagicMock

from chatqec.api import Answer, ChatQEC
from chatqec.conversation.classifier import TopicVerdict
from chatqec.conversation.memory import ConversationMemory
from chatqec.conversation.rewriter import RewriteResult
from chatqec.conversation.synthesizer import SynthesisResult
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk


def _retrieved_chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"x#{i}",
        source_id="x",
        text=f"c{i}",
        title=f"T{i}",
        authority_tier="preprint",
        score=0.5,
        payload={"url": "https://example.com", "text": f"c{i}"},
    )


def _build_chatqec(on_topic: bool = True, syn_text: str = "Answer [^1]."):
    classifier = MagicMock()
    classifier.classify.return_value = TopicVerdict(
        on_topic=on_topic,
        refusal_message="off topic" if not on_topic else "",
    )
    rewriter = MagicMock()
    rewriter.rewrite.return_value = RewriteResult(rewritten="q", variants=[])

    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        chunks=[_retrieved_chunk(0)],
        matched_topics=[],
    )
    canonical_index = MagicMock()
    canonical_index.pages = []

    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = SynthesisResult(
        text=syn_text,
        citations=[],
        confidence="medium",
    )
    return (
        ChatQEC(
            classifier=classifier,
            rewriter=rewriter,
            retriever=retriever,
            synthesizer=synthesizer,
            canonical_index=canonical_index,
            memory=ConversationMemory(),
        ),
        retriever,
        synthesizer,
    )


def test_ask_returns_answer_object_for_on_topic_question():
    bot, _, _ = _build_chatqec()
    answer = bot.ask("What is the surface code?")
    assert isinstance(answer, Answer)
    assert "Answer" in answer.text
    assert answer.confidence == "medium"


def test_ask_refuses_off_topic_question_without_calling_retriever():
    bot, retriever, synthesizer = _build_chatqec(on_topic=False)
    answer = bot.ask("Capital of France?")
    assert "off topic" in answer.text.lower() or "off_topic" in answer.text.lower()
    retriever.retrieve.assert_not_called()
    synthesizer.synthesize.assert_not_called()
    assert answer.confidence == "low"


def test_ask_records_turn_in_memory_for_followups():
    bot, _, _ = _build_chatqec()
    bot.ask("q1")
    bot.ask("q2")
    assert len(list(bot.memory.turns())) == 4


def test_ask_stream_yields_text_chunks():
    classifier = MagicMock()
    classifier.classify.return_value = TopicVerdict(on_topic=True)
    rewriter = MagicMock()
    rewriter.rewrite.return_value = RewriteResult(rewritten="q")
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        chunks=[_retrieved_chunk(0)],
        matched_topics=[],
    )
    canonical_index = MagicMock()
    canonical_index.pages = []
    synthesizer = MagicMock()
    synthesizer.synthesize_stream.return_value = iter(["The ", "surface ", "code."])
    synthesizer.synthesize.return_value = SynthesisResult(
        text="The surface code.", citations=[], confidence="low"
    )

    bot = ChatQEC(
        classifier=classifier,
        rewriter=rewriter,
        retriever=retriever,
        synthesizer=synthesizer,
        canonical_index=canonical_index,
        memory=ConversationMemory(),
    )
    chunks = list(bot.ask_stream("What is the surface code?"))
    assert "surface" in "".join(c.delta for c in chunks)


def test_ask_attaches_retrieved_figures_to_synthesizer(tmp_path):
    from PIL import Image

    img_path = tmp_path / "fig.png"
    Image.new("RGB", (4, 4), color="white").save(img_path)

    classifier = MagicMock()
    classifier.classify.return_value = TopicVerdict(on_topic=True)
    rewriter = MagicMock()
    rewriter.rewrite.return_value = RewriteResult(rewritten="q", variants=[])
    retriever = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="x#0",
        source_id="x",
        text="c0",
        title="T0",
        authority_tier="preprint",
        score=0.5,
        payload={"url": "https://example.com", "figures": [str(img_path)]},
    )
    retriever.retrieve.return_value = RetrievalResult(chunks=[chunk], matched_topics=[])
    canonical_index = MagicMock()
    canonical_index.pages = []
    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = SynthesisResult(
        text="Answer.", citations=[], confidence="medium"
    )

    bot = ChatQEC(
        classifier=classifier,
        rewriter=rewriter,
        retriever=retriever,
        synthesizer=synthesizer,
        canonical_index=canonical_index,
        memory=ConversationMemory(),
    )
    bot.ask("What is the surface code?")

    _, kwargs = synthesizer.synthesize.call_args
    assert len(kwargs["images"]) == 1


def test_ask_surfaces_tool_calls_when_synthesizer_invoked_tool():
    from chatqec.conversation.synthesizer import SynthesisResult
    from chatqec.conversation.tool_loop import ToolInvocation

    bot, _, synthesizer = _build_chatqec()
    synthesizer.synthesize.return_value = SynthesisResult(
        text="With tool result [^1].",
        citations=[],
        confidence="medium",
        tool_invocations=[
            ToolInvocation(
                name="stim_simulate",
                input={"circuit": "M 0\n", "shots": 1},
                output={"ok": True, "summary": "shape=(1,1)"},
            )
        ],
    )
    answer = bot.ask("Run M 0 once.")
    assert len(answer.tool_calls) == 1
    assert answer.tool_calls[0].name == "stim_simulate"
