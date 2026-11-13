"""Tool-use loop integration tests with mocked Anthropic SDK + MCP adapter."""

from unittest.mock import MagicMock

from chatqec.conversation.mcp_adapter import MCPAdapter, MCPToolDef
from chatqec.conversation.tool_loop import run_tool_loop


def _adapter_with_stim():
    fake = MagicMock()
    fake.list_tools.return_value = [
        MCPToolDef(
            name="stim_simulate",
            description="run",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    fake.call_tool.return_value = {"ok": True, "summary": "shape=(1,1)"}
    return MCPAdapter(client=fake)


def _msg(content_blocks, stop_reason="end_turn"):
    m = MagicMock()
    m.content = content_blocks
    m.stop_reason = stop_reason
    m.usage = MagicMock(input_tokens=10, output_tokens=20)
    return m


def _text_block(text):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name, args, tool_id="t1"):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = args
    b.id = tool_id
    return b


def test_loop_returns_immediately_when_no_tool_use():
    client = MagicMock()
    client.messages.create.return_value = _msg(
        [_text_block("simple answer")], stop_reason="end_turn"
    )
    final, calls = run_tool_loop(
        client=client,
        model="claude-sonnet-4-6",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        adapter=_adapter_with_stim(),
        max_iterations=3,
        max_tokens=500,
    )
    assert "simple answer" in final
    assert calls == []
    assert client.messages.create.call_count == 1


def test_loop_executes_tool_and_resumes():
    client = MagicMock()
    client.messages.create.side_effect = [
        _msg(
            [_tool_use_block("stim_simulate", {"circuit": "M 0\n", "shots": 1})],
            stop_reason="tool_use",
        ),
        _msg([_text_block("After tool: result is 0")], stop_reason="end_turn"),
    ]
    final, calls = run_tool_loop(
        client=client,
        model="claude-sonnet-4-6",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        adapter=_adapter_with_stim(),
        max_iterations=3,
        max_tokens=500,
    )
    assert "After tool" in final
    assert len(calls) == 1
    assert calls[0].name == "stim_simulate"
    assert calls[0].output["ok"] is True


def test_loop_caps_tool_rounds_then_forces_a_final_answer():
    # A model that always wants the tool when tools are offered; the loop must
    # cap tool rounds at max_iterations, then synthesize a real answer via one
    # final tool-free call instead of looping forever or returning a placeholder.
    client = MagicMock()

    def _create(*, model, max_tokens, system, messages, tools=None):
        if tools:
            return _msg(
                [_tool_use_block("stim_simulate", {"circuit": "M 0\n", "shots": 1})],
                stop_reason="tool_use",
            )
        return _msg([_text_block("synthesized after cap")], stop_reason="end_turn")

    client.messages.create.side_effect = _create

    final, calls = run_tool_loop(
        client=client,
        model="claude-sonnet-4-6",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        adapter=_adapter_with_stim(),
        max_iterations=2,
        max_tokens=500,
    )
    assert len(calls) == 2  # tool rounds capped at max_iterations
    assert final == "synthesized after cap"
    assert "stopped" not in final.lower()


def test_loop_handles_unknown_tool_gracefully():
    adapter = _adapter_with_stim()
    client = MagicMock()
    client.messages.create.side_effect = [
        _msg([_tool_use_block("nonexistent_tool", {})], stop_reason="tool_use"),
        _msg([_text_block("done")], stop_reason="end_turn"),
    ]
    final, calls = run_tool_loop(
        client=client,
        model="claude-sonnet-4-6",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        adapter=adapter,
        max_iterations=3,
        max_tokens=500,
    )
    assert "done" in final
    assert len(calls) == 1
    assert calls[0].output.get("error") is not None


def test_serialized_tool_result_drops_inline_svg_but_keeps_data():
    import json

    from chatqec.conversation.tool_loop import _serialize_tool_result

    svg = "<svg>" + "x" * 30_000 + "</svg>"
    out = json.loads(
        _serialize_tool_result(
            {"ok": True, "physical_qubits": ["QB5", "QB6"], "coupling_map_svg": svg, "note_svg": ""}
        )
    )
    assert out["physical_qubits"] == ["QB5", "QB6"]
    assert "<svg" not in out["coupling_map_svg"]
    assert "shown to the user" in out["coupling_map_svg"]
    assert out["note_svg"] == ""  # empty strings pass through untouched
