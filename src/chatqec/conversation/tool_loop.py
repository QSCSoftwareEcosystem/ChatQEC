"""Anthropic-native tool-use loop.

Repeatedly calls ``messages.create(tools=...)``. When the model emits a
``tool_use`` content block, invoke the tool via the MCP adapter, append a
``tool_result`` message, and re-call. Stop on ``end_turn`` or when
``max_iterations`` is hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chatqec.conversation.mcp_adapter import MCPAdapter


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


def _serialize_tool_result(output: dict[str, Any]) -> str:
    return json.dumps(output)


def run_tool_loop(
    *,
    client: Any,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    adapter: MCPAdapter,
    max_iterations: int = 5,
    max_tokens: int = 4000,
) -> tuple[str, list[ToolInvocation]]:
    """Drive the Anthropic messages API through tool-use cycles.

    Returns ``(final_assistant_text, list_of_tool_invocations)``.
    """
    invocations: list[ToolInvocation] = []
    convo: list[dict[str, Any]] = list(messages)
    tool_schemas = adapter.tool_schemas_for_sdk()

    for _ in range(max_iterations):
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=convo,
            tools=tool_schemas,
        )

        if msg.stop_reason != "tool_use":
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            return str(text), invocations

        convo.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": getattr(b, "text", "")}
                    if getattr(b, "type", "") == "text"
                    else {
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    }
                    for b in msg.content
                ],
            }
        )

        tool_results: list[dict[str, Any]] = []
        for b in msg.content:
            if getattr(b, "type", "") != "tool_use":
                continue
            try:
                result = adapter.invoke(b.name, b.input)
            except KeyError:
                result = {"error": f"unknown tool: {b.name}"}
            invocations.append(ToolInvocation(name=b.name, input=dict(b.input), output=result))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": _serialize_tool_result(result),
                }
            )
        convo.append({"role": "user", "content": tool_results})

    return (
        "(stopped after reaching tool-use iteration limit; " "partial results in tool_calls)",
        invocations,
    )
