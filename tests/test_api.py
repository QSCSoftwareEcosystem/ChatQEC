"""Smoke tests for public dataclasses (Answer, Citation).

Heavy components (Anthropic, Qdrant) are not exercised here — see
test_api_chatqec.py for end-to-end pipeline tests with mocks.
"""

import dataclasses

import pytest

from chatqec.api import Answer, Citation


def test_answer_is_immutable():
    a = Answer(text="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.text = "x"  # type: ignore[misc]


def test_answer_defaults():
    a = Answer(text="hello")
    assert a.citations == []
    assert a.tool_calls == []
    assert a.confidence == "low"


def test_citation_is_immutable():
    c = Citation(source_id="s", chunk_id="c", title="t", url="u")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.title = "x"  # type: ignore[misc]
