"""Query rewriter — resolves pronouns, generates retrieval variants."""

import json
from unittest.mock import MagicMock

from chatqec.conversation.memory import ConversationMemory
from chatqec.conversation.rewriter import QueryRewriter, RewriteResult


def _client_returning(json_text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=json_text)]
    client.messages.create.return_value = msg
    return client


def test_rewriter_returns_self_contained_query_and_variants():
    client = _client_returning(
        json.dumps(
            {
                "rewritten": "What is the threshold for the surface code?",
                "variants": [
                    "What is the surface code threshold under depolarizing noise?",
                    "What is the fault-tolerant threshold for the surface code?",
                ],
                "is_followup": False,
            }
        )
    )
    rw = QueryRewriter(client=client, model="claude-haiku-4-5")
    memory = ConversationMemory()
    result = rw.rewrite("What's the threshold?", memory=memory)
    assert isinstance(result, RewriteResult)
    assert "surface code" in result.rewritten.lower()
    assert len(result.variants) == 2


def test_rewriter_resolves_pronoun_from_recent_turn():
    client = _client_returning(
        json.dumps(
            {
                "rewritten": "What is the surface code threshold?",
                "variants": [],
                "is_followup": True,
            }
        )
    )
    rw = QueryRewriter(client=client, model="claude-haiku-4-5")
    memory = ConversationMemory()
    memory.add_user("Tell me about the surface code.")
    memory.add_assistant("It's a topological CSS code with high threshold.")
    result = rw.rewrite("What's its threshold?", memory=memory)
    assert result.is_followup is True
    call_kwargs = client.messages.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    user_msg = next(m for m in sent_messages if m["role"] == "user")
    assert "surface code" in user_msg["content"].lower()


def test_rewriter_treats_malformed_json_as_passthrough():
    client = _client_returning("not json")
    rw = QueryRewriter(client=client, model="claude-haiku-4-5")
    memory = ConversationMemory()
    result = rw.rewrite("raw query", memory=memory)
    assert result.rewritten == "raw query"
    assert result.variants == []
    assert result.is_followup is False


def test_rewriter_first_turn_query_passes_through_unchanged():
    client = _client_returning(
        json.dumps(
            {
                "rewritten": "What is the surface code?",
                "variants": [],
                "is_followup": False,
            }
        )
    )
    rw = QueryRewriter(client=client, model="claude-haiku-4-5")
    memory = ConversationMemory()
    result = rw.rewrite("What is the surface code?", memory=memory)
    assert result.is_followup is False
