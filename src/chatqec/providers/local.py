"""Local OpenAI-compatible LLM client (vLLM / ``transformers serve``).

Speaks the Anthropic ``client.messages.create(...)`` shape, including the
tool-use loop, against a self-hosted ``/v1/chat/completions`` endpoint. Used to
test chatQEC end to end on local open-weights models (Muse-Glimmer-30B, Gemma 4)
without spending Claude or Gemini tokens; see ``scripts/serve_local_models.sh``.

Translation (Anthropic -> OpenAI):
  - ``tools`` ``{name, description, input_schema}`` -> ``{type: function, function: {..., parameters}}``
  - assistant ``tool_use`` blocks -> ``tool_calls`` (arguments as a JSON string)
  - user ``tool_result`` blocks -> one ``role: tool`` message each
  - response ``tool_calls`` -> ``ToolUseBlock`` with ``stop_reason="tool_use"``

The server owns the model-specific tool-call format (e.g. Muse's ``atem`` XML)
through its tool parser, and its reasoning parser strips thinking out of
``content``. Thinking still consumes ``max_tokens``, hence ``reasoning_headroom``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from chatqec.providers.base import MessageResponse, TextBlock, ToolUseBlock, Usage
from chatqec.providers.huggingface import _anthropic_to_oai_messages


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return json.dumps(content)


def _to_oai_messages(*, system: str | None, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Like ``_anthropic_to_oai_messages`` but also carries tool-use turns."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        blocks = content if isinstance(content, list) else []
        if not any(b.get("type") in ("tool_use", "tool_result") for b in blocks):
            out.extend(_anthropic_to_oai_messages(system=None, messages=[m]))
            continue
        if m["role"] == "assistant":
            text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            out.append({"role": "assistant", "content": text or None, "tool_calls": calls})
            continue
        for b in blocks:
            if b.get("type") == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": _tool_result_text(b.get("content")),
                    }
                )
            elif b.get("type") == "text":
                out.append({"role": "user", "content": str(b.get("text", ""))})
    return out


def _to_oai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _from_oai_response(body: dict[str, Any]) -> MessageResponse:
    msg = body["choices"][0]["message"]
    blocks: list[TextBlock | ToolUseBlock] = []
    if msg.get("content"):
        blocks.append(TextBlock(text=msg["content"]))
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        blocks.append(
            ToolUseBlock(
                id=tc.get("id") or f"local_call_{i}",
                name=fn.get("name", ""),
                input=_parse_arguments(fn.get("arguments")),
            )
        )
    usage = body.get("usage") or {}
    return MessageResponse(
        content=blocks,
        stop_reason="tool_use" if any(isinstance(b, ToolUseBlock) for b in blocks) else "end_turn",
        usage=Usage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        ),
    )


class _MessagesProxy:
    def __init__(self, client: LocalLLM) -> None:
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
        payload: dict[str, Any] = {
            "model": model,
            # Callers size max_tokens for the visible answer (the classifier asks
            # for ~10); reasoning models spend tokens thinking first, so add room.
            "max_tokens": max_tokens + self._client.reasoning_headroom,
            "messages": _to_oai_messages(system=system, messages=messages or []),
        }
        if self._client.reasoning_strength:
            # Muse-Glimmer's template reads this kwarg; templates without it ignore it.
            payload["chat_template_kwargs"] = {
                "reasoning_strength": self._client.reasoning_strength
            }
        if tools:
            payload["tools"] = _to_oai_tools(tools)
        where = self._client.base_url
        try:
            resp = self._client._http.post("/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            # "local llm unavailable" is a recoverable marker in FallbackLLMClient,
            # so a dead primary server falls through to the next local model.
            raise RuntimeError(f"local LLM unavailable at {where}: {e}") from e
        if resp.is_server_error:
            raise RuntimeError(
                f"local LLM unavailable at {where} ({resp.status_code}): {resp.text[:300]}"
            )
        if resp.is_client_error:
            raise RuntimeError(
                f"local LLM rejected request at {where} ({resp.status_code}): {resp.text[:300]}"
            )
        return _from_oai_response(resp.json())


class LocalLLM:
    """Anthropic-shaped client backed by a local OpenAI-compatible server."""

    def __init__(
        self,
        *,
        base_url: str,
        reasoning_strength: str | None = None,
        reasoning_headroom: int = 2048,
        timeout: float = 600.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.reasoning_strength = reasoning_strength
        self.reasoning_headroom = reasoning_headroom
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)
        self.messages = _MessagesProxy(self)

    def is_up(self) -> bool:
        """True when the server answers ``GET /models`` (cheap liveness probe)."""
        try:
            return self._http.get("/models", timeout=3.0).is_success
        except httpx.HTTPError:
            return False
