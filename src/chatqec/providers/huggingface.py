"""HuggingFace Inference API LLM client.

Wraps ``huggingface_hub.InferenceClient`` so it speaks the Anthropic
``client.messages.create(...)`` shape used by the classifier, rewriter,
synthesizer, and eval judge.

Defaults to US-approved open-weights models (Llama family, GPT-OSS); see
``conf/config.yaml`` ``models.hf.*`` to override per-role.

Limitations vs. Anthropic provider:
  - Tool-use loop is NOT supported. Pass ``mcp_adapter=None`` to Synthesizer.
  - ``messages.stream(...)`` falls through to a synchronous create + single-yield;
    no token-by-token streaming on this provider in v0.1.x.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from chatqec.providers.base import MessageResponse, TextBlock, Usage


def _anthropic_to_oai_messages(
    *, system: str | None, messages: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Translate Anthropic-style call args into OpenAI-style chat messages."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        else:
            text_parts = [str(b.get("text", "")) for b in content if b.get("type") == "text"]
            out.append({"role": role, "content": "\n".join(text_parts)})
    return out


class _StreamContext:
    """Minimal context manager mimicking anthropic stream's ``.text_stream``."""

    def __init__(self, text: str) -> None:
        self.text_stream: Iterator[str] = iter([text])

    def __enter__(self) -> _StreamContext:
        return self

    def __exit__(self, *exc: object) -> None: ...


class _MessagesProxy:
    def __init__(self, client: HuggingFaceLLM) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> MessageResponse:
        if tools:
            raise NotImplementedError(
                "HuggingFace provider does not yet support Anthropic-style tool use. "
                "Use the anthropic provider when --mcp-server is enabled."
            )
        oai_messages = _anthropic_to_oai_messages(system=system, messages=messages or [])
        try:
            resp = self._client._raw.chat.completions.create(
                model=model,
                messages=oai_messages,
                max_tokens=max_tokens,
            )
        except Exception as e:
            msg = str(e)
            if "403" in msg or "Forbidden" in msg or "permissions" in msg:
                raise RuntimeError(
                    "HuggingFace returned 403. Your HF_TOKEN likely lacks the "
                    "'Make calls to Inference Providers' permission, or you have "
                    "exhausted the monthly free credit. "
                    "Create a fine-grained token at "
                    "https://huggingface.co/settings/tokens/new?tokenType=fineGrained "
                    "with the Inference Providers scope ticked."
                ) from e
            raise
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return MessageResponse(
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
        )

    @contextmanager
    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Iterator[_StreamContext]:
        resp = self.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
        first = resp.content[0] if resp.content else None
        text = first.text if isinstance(first, TextBlock) else ""
        yield _StreamContext(text=text)


class HuggingFaceLLM:
    """Anthropic-shaped client backed by HuggingFace's serverless inference."""

    def __init__(self, *, api_key: str | None = None, provider: str = "auto") -> None:
        from huggingface_hub import InferenceClient

        token = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        self._raw = InferenceClient(provider=provider, api_key=token)  # type: ignore[arg-type]
        self.messages = _MessagesProxy(self)
