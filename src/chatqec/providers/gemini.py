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
                if btype == "image":
                    src = b.get("source", {})
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": src.get("media_type", ""),
                                "data": src.get("data", ""),
                            }
                        }
                    )
                elif btype == "text":
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


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local JSON-pointer ``$ref`` like ``#/$defs/Name`` against root.

    Returns None for anything not a simple local pointer (e.g. an external
    URI) — the caller degrades to an untyped field rather than crashing.
    """
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _sanitize_schema(schema: Any, *, root: dict[str, Any] | None = None) -> Any:
    """Strip/translate JSON-schema constructs Gemini's function declarations
    reject outright (``ValueError: Unknown field for Schema: <key>``) rather
    than ignore.

    Two constructs need real handling, not just dropping a key:

    - ``anyOf``: Pydantic renders every ``X | None`` field (e.g. ``rounds:
      int | None`` in threshold_sweep's input model) as
      ``{"anyOf": [{...X...}, {"type": "null"}]}``. For that common
      Optional[X] shape — exactly one non-null variant — collapse to the
      variant's own schema so type info survives. A true multi-type union
      has no single equivalent Gemini schema, so it degrades to an untyped
      field.
    - ``$ref``/``$defs``: a tool with nested Pydantic submodels (e.g.
      glcb_visualize_url's list-of-gate-objects fields) renders each nested
      type once under a top-level ``$defs`` map and points to it via
      ``$ref: "#/$defs/Name"``. Gemini's Schema proto has neither ``$ref``
      nor ``$defs``, so each reference is resolved and inlined here (using
      the original top-level schema as ``root`` throughout the recursion,
      since ``$ref`` paths are always relative to the document root, not to
      whichever nested dict is being processed) before ``$defs`` itself is
      dropped as unused.
    """
    if isinstance(schema, dict):
        root = schema if root is None else root
        if "$ref" in schema:
            resolved = _resolve_ref(schema["$ref"], root)
            if resolved is None:
                return {}
            merged = {**{k: v for k, v in schema.items() if k != "$ref"}, **resolved}
            return _sanitize_schema(merged, root=root)
        if "anyOf" in schema:
            variants = [v for v in schema["anyOf"] if v.get("type") != "null"]
            schema = {k: v for k, v in schema.items() if k != "anyOf"}
            if len(variants) == 1:
                schema = {**schema, **variants[0]}
        return {
            k: _sanitize_schema(v, root=root)
            for k, v in schema.items()
            if k not in _GEMINI_SCHEMA_DROP
        }
    if isinstance(schema, list):
        return [_sanitize_schema(v, root=root) for v in schema]
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
