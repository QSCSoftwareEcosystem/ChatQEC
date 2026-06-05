"""Public ChatQEC class — stub returning a placeholder until M7."""

from chatqec.api import ChatQEC, Answer


def test_chatqec_ask_returns_answer_object():
    bot = ChatQEC()
    answer = bot.ask("What is the surface code?")
    assert isinstance(answer, Answer)
    assert answer.text  # non-empty
    assert answer.citations == []
    assert answer.confidence in {"low", "medium", "high"}


def test_answer_is_immutable():
    bot = ChatQEC()
    answer = bot.ask("foo")
    # dataclass(frozen=True) raises on field assignment
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "changed"  # type: ignore[misc]
