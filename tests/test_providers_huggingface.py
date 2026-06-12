"""HuggingFace LLM provider — Anthropic-shaped wrapper around InferenceClient."""

from unittest.mock import MagicMock

import pytest

from chatqec.providers.huggingface import HuggingFaceLLM


def _stub_raw_client(text: str, prompt_tokens: int = 12, completion_tokens: int = 7):
    """Return an object mimicking InferenceClient.chat.completions.create's return."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_llm(text: str = "ok") -> HuggingFaceLLM:
    llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
    llm._raw = MagicMock()
    llm._raw.chat.completions.create.return_value = _stub_raw_client(text)
    from chatqec.providers.huggingface import _MessagesProxy

    llm.messages = _MessagesProxy(llm)
    return llm


def test_create_returns_anthropic_shaped_response():
    llm = _make_llm("on_topic")
    resp = llm.messages.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=10,
        system="You are a classifier.",
        messages=[{"role": "user", "content": "Is QEC interesting?"}],
    )
    assert resp.content[0].text == "on_topic"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 7


def test_create_translates_system_and_messages_into_oai_chat():
    llm = _make_llm("ok")
    llm.messages.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=200,
        system="be precise",
        messages=[
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
            {"role": "user", "content": "follow up"},
        ],
    )
    sent = llm._raw.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "be precise"}
    assert sent[1] == {"role": "user", "content": "prev q"}
    assert sent[-1] == {"role": "user", "content": "follow up"}


def test_create_handles_anthropic_content_blocks():
    llm = _make_llm("done")
    llm.messages.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=50,
        system=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "block A"},
                    {"type": "text", "text": "block B"},
                ],
            }
        ],
    )
    sent = llm._raw.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["role"] == "user"
    assert "block A" in sent[0]["content"]
    assert "block B" in sent[0]["content"]


def test_create_raises_on_tool_use():
    llm = _make_llm("won't reach")
    with pytest.raises(NotImplementedError):
        llm.messages.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            max_tokens=50,
            system="sys",
            messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "stim_simulate", "input_schema": {}}],
        )


def test_stream_yields_full_text_once():
    llm = _make_llm("hello world")
    with llm.messages.stream(
        model="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=200,
        system=None,
        messages=[{"role": "user", "content": "say hi"}],
    ) as stream:
        chunks = list(stream.text_stream)
    assert chunks == ["hello world"]


def test_create_returns_empty_text_on_empty_response():
    llm = _make_llm("")
    llm._raw.chat.completions.create.return_value.choices[0].message.content = None
    resp = llm.messages.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=10,
        system="sys",
        messages=[{"role": "user", "content": "q"}],
    )
    assert resp.content[0].text == ""
