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
from chatqec.trace import trace


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


def _serialize_tool_result(output: dict[str, Any]) -> str:
    """The tool result as the model sees it.

    Inline SVG fields (``coupling_map_svg``, ``plot_svg``, ...) are replaced by a short
    note: the UI renders them from the full recorded output, and the model cannot use the
    markup. A 20-qubit coupling map is ~23 KB (~8k tokens), so two mapping calls
    overflowed the local models' 32k context.
    """
    slim = {
        k: (
            f"[SVG diagram, {len(v)} chars, shown to the user by the UI]"
            if k.endswith("_svg") and isinstance(v, str) and "<svg" in v
            else v
        )
        for k, v in output.items()
    }
    return json.dumps(slim)


def run_tool_loop(
    *,
    client: Any,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    adapter: MCPAdapter,
    max_iterations: int = 8,
    max_tokens: int = 4000,
) -> tuple[str, list[ToolInvocation]]:
    """Drive the Anthropic messages API through tool-use cycles.

    Returns ``(final_assistant_text, list_of_tool_invocations)``.
    """
    invocations: list[ToolInvocation] = []
    convo: list[dict[str, Any]] = list(messages)
    tool_schemas = adapter.tool_schemas_for_sdk()
    trace(
        "tool_loop.start",
        model=model,
        max_iterations=max_iterations,
        tools=[t.get("name") for t in tool_schemas],
    )

    for iteration in range(max_iterations):
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=convo,
            tools=tool_schemas,
        )
        trace("tool_loop.iteration", iteration=iteration, stop_reason=msg.stop_reason)

        if msg.stop_reason != "tool_use":
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            trace("tool_loop.final", iteration=iteration, n_tool_calls=len(invocations))
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
            trace("tool_loop.tool_use", name=b.name, input=dict(b.input))
            try:
                result = adapter.invoke(b.name, b.input)
            except KeyError:
                result = {"error": f"unknown tool: {b.name}"}
            trace("tool_loop.tool_result", name=b.name, output=result)
            invocations.append(ToolInvocation(name=b.name, input=dict(b.input), output=result))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": _serialize_tool_result(result),
                }
            )
        convo.append({"role": "user", "content": tool_results})

    # Hit the iteration cap with the model still requesting tools. Force a final
    # answer by calling once more WITHOUT tools, so it must synthesize from the
    # tool results gathered so far instead of looping forever.
    trace("tool_loop.forced_final", n_tool_calls=len(invocations), max_iterations=max_iterations)
    final = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=convo,
    )
    text = "".join(b.text for b in final.content if getattr(b, "type", "") == "text")
    return str(text), invocations
