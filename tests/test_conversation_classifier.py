"""Off-topic classifier — cheap Haiku-based pre-filter for cost protection."""

from unittest.mock import MagicMock

from chatqec.conversation.classifier import OffTopicClassifier, TopicVerdict


def _make_anthropic_with_response(text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client.messages.create.return_value = msg
    return client


def test_classifier_returns_on_topic_for_qec_query():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("What is the surface code threshold?")
    assert isinstance(verdict, TopicVerdict)
    assert verdict.on_topic is True


def test_classifier_returns_off_topic_for_unrelated_query():
    client = _make_anthropic_with_response("off_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("What's the capital of France?")
    assert verdict.on_topic is False
    assert verdict.refusal_message


def test_classifier_treats_malformed_response_as_off_topic_to_be_safe():
    client = _make_anthropic_with_response("not a clean label")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("ambiguous")
    assert verdict.on_topic is False


def test_classifier_includes_model_name_in_api_call():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("q")
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
    assert call_kwargs["max_tokens"] <= 20
