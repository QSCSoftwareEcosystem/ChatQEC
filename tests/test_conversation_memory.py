"""In-memory conversation log used to feed the rewriter and synthesizer."""

import dataclasses

import pytest

from chatqec.conversation.memory import ConversationMemory, Turn


def test_memory_records_turns_in_order():
    m = ConversationMemory()
    m.add_user("What is the surface code?")
    m.add_assistant("It's a topological CSS code.")
    m.add_user("And the threshold?")
    log = list(m.turns())
    assert [t.role for t in log] == ["user", "assistant", "user"]
    assert log[0].content == "What is the surface code?"


def test_memory_recent_returns_last_n_turns_capped():
    m = ConversationMemory()
    for i in range(10):
        m.add_user(f"q{i}")
        m.add_assistant(f"a{i}")
    last4 = m.recent(n=4)
    assert len(last4) == 4
    assert last4[-1].content == "a9"


def test_memory_empty_returns_empty_recent():
    m = ConversationMemory()
    assert m.recent(n=5) == []


def test_turn_is_immutable():
    t = Turn(role="user", content="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.content = "y"  # type: ignore[misc]


def test_memory_clear_empties_history():
    m = ConversationMemory()
    m.add_user("x")
    m.add_assistant("y")
    m.clear()
    assert list(m.turns()) == []
