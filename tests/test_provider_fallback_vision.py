"""Fallback vision-model routing for image requests."""

from unittest.mock import MagicMock

from chatqec.providers.base import MessageResponse, TextBlock
from chatqec.providers.fallback import Backend, FallbackLLMClient


def test_image_request_uses_vision_model():
    inner = MagicMock()
    inner.messages.create.return_value = MessageResponse(content=[TextBlock(text="ok")])
    be = Backend(name="hf", client=inner, default_model="text-model", vision_model="vision-model")
    client = FallbackLLMClient(backends=[be], verbose=False)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
    )
    assert inner.messages.create.call_args.kwargs["model"] == "vision-model"


def test_text_request_uses_default_model():
    inner = MagicMock()
    inner.messages.create.return_value = MessageResponse(content=[TextBlock(text="ok")])
    be = Backend(name="hf", client=inner, default_model="text-model", vision_model="vision-model")
    client = FallbackLLMClient(backends=[be], verbose=False)

    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert inner.messages.create.call_args.kwargs["model"] == "text-model"
