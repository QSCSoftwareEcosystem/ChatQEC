"""Anthropic-shaped Gemini LLM client with tool-use translation.

Lets Gemini stand in for Anthropic in the ``FallbackLLMClient`` chain — including
the MCP tool-use loop. ``messages.create(...)`` accepts Anthropic-style messages
and tool schemas, translates them to Gemini's function-calling format, and maps
the response back to an Anthropic-shaped ``MessageResponse`` (with
``ToolUseBlock`` when Gemini requests a function call).

The translation helpers are pure and unit-tested; the real ``genai`` model
factory is a thin wrapper exercised end-to-end.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from chatqec.providers.base import MessageResponse, TextBlock, ToolUseBlock, Usage

ModelFactory = Callable[..., Any]


def _tool_use_id_to_name(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map each assistant ``tool_use`` id to its tool name (for tool_result)."""
    mapping: dict[str, str] = {}
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "tool_use":
                mapping[b["id"]] = b["name"]
    return mapping


def _parse_tool_result(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"result": content}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    if isinstance(content, dict):
        return content
    return {"result": content}


def _to_gemini_contents(
    *, system: str | None, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate Anthropic-style messages into Gemini ``contents``.

    ``system`` is handled separately (as ``system_instruction``), not inlined.
    """
    id_to_name = _tool_use_id_to_name(messages)
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        content = m["content"]
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for b in content:
                btype = b.get("type")
                if btype == "text":
                    parts.append({"text": b.get("text", "")})
                elif btype == "tool_use":
                    parts.append(
                        {"function_call": {"name": b["name"], "args": dict(b.get("input", {}))}}
                    )
                elif btype == "tool_result":
                    name = id_to_name.get(b.get("tool_use_id", ""), "")
                    parts.append(
                        {
                            "function_response": {
                                "name": name,
                                "response": _parse_tool_result(b.get("content")),
                            }
                        }
                    )
        contents.append({"role": role, "parts": parts})
    return contents


def _from_gemini_response(resp: Any) -> MessageResponse:
    """Map a Gemini response into an Anthropic-shaped ``MessageResponse``."""
    parts = resp.candidates[0].content.parts
    blocks: list[TextBlock | ToolUseBlock] = []
    saw_tool = False
    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            saw_tool = True
            args = dict(getattr(fc, "args", {}) or {})
            blocks.append(ToolUseBlock(id=f"gemini_call_{i}", name=fc.name, input=args))
            continue
        text = getattr(part, "text", None)
        if text:
            blocks.append(TextBlock(text=text))
    return MessageResponse(
        content=blocks,
        stop_reason="tool_use" if saw_tool else "end_turn",
        usage=Usage(),
    )


# JSON-schema keywords Gemini's FunctionDeclaration Schema proto rejects.
_GEMINI_SCHEMA_DROP = frozenset(
    {
        "title",
        "additionalProperties",
        "$defs",
        "$schema",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "const",
        "examples",
    }
)


def _sanitize_schema(schema: Any) -> Any:
    """Strip JSON-schema keys Gemini's function declarations reject."""
    if isinstance(schema, dict):
        return {k: _sanitize_schema(v) for k, v in schema.items() if k not in _GEMINI_SCHEMA_DROP}
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


def _default_model_factory(
    *, model: str, tools: Any, system: str | None
) -> Any:  # pragma: no cover
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    gem_tools = None
    if tools:
        gem_tools = [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": _sanitize_schema(t.get("input_schema", {})),
                    }
                    for t in tools
                ]
            }
        ]
    return genai.GenerativeModel(model_name=model, tools=gem_tools, system_instruction=system)


class _MessagesProxy:
    def __init__(self, factory: ModelFactory) -> None:
        self._factory = factory

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> MessageResponse:
        contents = _to_gemini_contents(system=system, messages=messages or [])
        gen_model = self._factory(model=model, tools=tools, system=system)
        resp = gen_model.generate_content(contents)
        return _from_gemini_response(resp)


class GeminiLLM:
    """Anthropic-shaped client backed by Google Gemini (function-calling)."""

    def __init__(self, *, model_factory: ModelFactory | None = None) -> None:
        self.messages = _MessagesProxy(model_factory or _default_model_factory)
