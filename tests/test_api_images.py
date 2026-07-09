from unittest.mock import MagicMock

from chatqec.api import ChatQEC
from chatqec.api_types import ImageAttachment


def _bot():
    classifier = MagicMock()
    classifier.classify.return_value = MagicMock(on_topic=True, refusal_message="")
    rewriter = MagicMock()
    rewriter.rewrite.return_value = MagicMock(rewritten="q", variants=[], is_followup=False)
    rewriter.client = MagicMock()
    rewriter.model = "gemini-2.5-flash"
    retriever = MagicMock()
    retriever.retrieve.return_value = MagicMock(chunks=[], matched_topics=[], debug={})
    synth = MagicMock()
    synth.synthesize.return_value = MagicMock(
        text="ans",
        citations=[],
        confidence="low",
        tool_invocations=[],
        input_tokens=0,
        output_tokens=0,
    )
    canonical = MagicMock()
    canonical.pages = []
    bot = ChatQEC(
        classifier=classifier,
        rewriter=rewriter,
        retriever=retriever,
        synthesizer=synth,
        canonical_index=canonical,
    )
    return bot, synth, classifier, retriever


def test_ask_passes_images_to_synthesizer(monkeypatch):
    bot, synth, _, _ = _bot()
    monkeypatch.setattr("chatqec.api.describe_image", lambda **k: "a lattice figure")
    img = ImageAttachment(data=b"x", media_type="image/png")
    bot.ask("what is this?", images=[img])
    assert synth.synthesize.call_args.kwargs["images"] == [img]


def test_caption_feeds_classifier_and_retrieval(monkeypatch):
    bot, _, classifier, retriever = _bot()
    monkeypatch.setattr("chatqec.api.describe_image", lambda **k: "surface code lattice")
    img = ImageAttachment(data=b"x", media_type="image/png")
    bot.ask("explain", images=[img])
    # classifier sees question + caption
    assert "surface code lattice" in classifier.classify.call_args.args[0]
    # retrieval queries include the caption
    assert "surface code lattice" in retriever.retrieve.call_args.kwargs["queries"]


def test_no_caption_call_without_images(monkeypatch):
    bot, synth, _, _ = _bot()
    called = {"n": 0}
    monkeypatch.setattr(
        "chatqec.api.describe_image", lambda **k: called.__setitem__("n", called["n"] + 1) or ""
    )
    bot.ask("plain text question")
    assert called["n"] == 0
    assert synth.synthesize.call_args.kwargs["images"] == []
