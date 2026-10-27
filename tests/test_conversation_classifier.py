"""Off-topic classifier — cheap Haiku-based pre-filter for cost protection."""

from unittest.mock import MagicMock

from chatqec.conversation.classifier import OffTopicClassifier, TopicVerdict
from chatqec.conversation.memory import ConversationMemory


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


def test_classifier_on_topic_when_model_appends_content_after_label():
    # Haiku sometimes emits the label then keeps answering (e.g. when the query
    # says "give me an example"): 'on_topic\n\nThe Steane code is ...'. The verdict
    # must key off the leading label, not exact-match the whole response.
    client = _make_anthropic_with_response("on_topic\n\nThe Steane code is a CSS code")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("what is steane code. Give me example")
    assert verdict.on_topic is True


def test_classifier_on_topic_tolerates_trailing_punctuation():
    client = _make_anthropic_with_response("on_topic.")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    assert clf.classify("what is the color code?").on_topic is True


def test_classifier_off_topic_when_label_precedes_content():
    client = _make_anthropic_with_response("off_topic\n\nThis is about cooking.")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    assert clf.classify("best pasta recipe").on_topic is False


def test_classifier_prompt_covers_stabilizer_formalism():
    # Regression guard: Haiku consistently mis-refused "stabilizer of state |1>"
    # because the prompt under-specified the stabilizer formalism. The system prompt
    # sent to the API must explicitly cover it (incl. the stabilizer of a state).
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("what is the stabilizer of state |1>")
    system = client.messages.create.call_args.kwargs["system"].lower()
    assert "stabilizer formalism" in system
    assert "|1>" in system  # a specific single-qubit state is named as on_topic


def test_classifier_prompt_covers_ibm_hardware_topics():
    # Regression guard: the IBM hardware-aware tools (backend calibration, circuit
    # mapping) were added after this prompt was written, and the classifier refused
    # plain hardware questions like "what IBM backends do I have access to" because
    # nothing in the prompt said they were on_topic. The system prompt sent to the
    # API must explicitly cover IBM hardware/calibration/mapping questions.
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("What IBM Quantum backends do I have access to right now?")
    system = client.messages.create.call_args.kwargs["system"].lower()
    assert "ibm" in system
    assert "calibration" in system or "coupling map" in system


def test_classifier_on_topic_for_ibm_backend_query():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("What's the T1 time of qubit 0 on ibm_fez?")
    assert verdict.on_topic is True


def test_classifier_on_topic_for_circuit_mapping_query():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    verdict = clf.classify("Map a Bell state onto ibm_fez")
    assert verdict.on_topic is True


def test_classifier_includes_model_name_in_api_call():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("q")
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
    assert call_kwargs["max_tokens"] <= 20


def test_classifier_without_memory_sends_bare_query_unchanged():
    # Backward compatibility: callers that never pass `memory` (or pass an
    # empty one) must see the exact prior behavior — a bare query, no
    # "Recent conversation:" wrapper.
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("what is the surface code?")
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content == "what is the surface code?"


def test_classifier_with_empty_memory_sends_bare_query_unchanged():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("what is the surface code?", memory=ConversationMemory())
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content == "what is the surface code?"


def test_classifier_includes_recent_history_when_memory_has_turns():
    # Regression guard: a context-dependent follow-up with no QEC keywords of
    # its own ("how many rounds you have considered?") was misclassified
    # off_topic because the classifier never saw the conversation it
    # continued. The API call must now include recent turns as context.
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    memory = ConversationMemory()
    memory.add_user("What's the logical error rate of the distance-5 rotated surface code?")
    memory.add_assistant("Here are the simulation results from 100,000 shots...")

    verdict = clf.classify("how many rounds you have considered?", memory=memory)

    assert verdict.on_topic is True
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Recent conversation:" in content
    assert "distance-5 rotated surface code" in content
    assert "New question: how many rounds you have considered?" in content


def test_classifier_history_respects_history_turns_limit():
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5", history_turns=2)
    memory = ConversationMemory()
    memory.add_user("first on-topic question about stabilizer codes")
    memory.add_assistant("first answer")
    memory.add_user("second question")
    memory.add_assistant("second answer")

    clf.classify("a follow-up", memory=memory)

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "first on-topic question" not in content
    assert "second question" in content
    assert "second answer" in content


def test_classifier_prompt_covers_fault_tolerant_compilation_and_vendor_tools():
    # Regression guard: "Compile this rotation to Clifford+T and give the T count" was
    # refused as off_topic in 2 of 3 local-model runs, so the ibm_ft_compile tool never ran.
    # Each tool-backed topic must be named in the prompt as on_topic.
    client = _make_anthropic_with_response("on_topic")
    clf = OffTopicClassifier(client=client, model="claude-haiku-4-5")
    clf.classify("Compile this one-qubit rotation to Clifford+T and give me the T count")
    system = client.messages.create.call_args.kwargs["system"].lower()
    for phrase in (
        "clifford+t",
        "t count",
        "gridsynth",
        "pauli-based",
        "pauli checks",
        "postselection",
        "deltakit",
        "guppy",
    ):
        assert phrase in system, phrase
