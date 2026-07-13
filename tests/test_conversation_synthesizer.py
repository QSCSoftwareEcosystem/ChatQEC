"""Synthesizer — wraps Anthropic SDK, supports non-streaming and streaming."""

from unittest.mock import MagicMock

from chatqec.conversation.memory import Turn
from chatqec.conversation.synthesizer import SynthesisResult, Synthesizer
from chatqec.retrieval.hybrid import RetrievedChunk


def _ctx(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"x#{i}",
        source_id="x",
        text=f"chunk {i}",
        title=f"T{i}",
        authority_tier="preprint",
        score=0.5,
        payload={"url": "https://example.com"},
    )


def _web_ctx(i: int) -> RetrievedChunk:
    sid = f"web:{i:08x}"
    return RetrievedChunk(
        chunk_id=f"{sid}#0",
        source_id=sid,
        text=f"web chunk {i}",
        title=f"W{i}",
        authority_tier="web",
        score=0.5,
        payload={"url": f"https://web.example.com/{i}"},
    )


def _client_with_text(text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text=text)]
    msg.usage = MagicMock(input_tokens=100, output_tokens=50)
    client.messages.create.return_value = msg
    return client


def test_synthesize_returns_answer_text_and_citations():
    client = _client_with_text("The surface code is great [^1]. Period.")
    synth = Synthesizer(client=client, model="claude-sonnet-4-6")
    result = synth.synthesize(
        question="What is the surface code?",
        retrieved=[_ctx(0)],
        canonical_chunks=[],
        history=[],
    )
    assert isinstance(result, SynthesisResult)
    assert "surface code" in result.text.lower()
    assert len(result.citations) == 1


def test_synthesize_passes_system_prompt_and_context_to_anthropic():
    client = _client_with_text("answer [^1]")
    synth = Synthesizer(client=client, model="claude-sonnet-4-6")
    synth.synthesize(question="q", retrieved=[_ctx(0)], canonical_chunks=[], history=[])
    kwargs = client.messages.create.call_args.kwargs
    assert "ChatQEC" in kwargs["system"]
    user_msg = kwargs["messages"][-1]
    assert user_msg["role"] == "user"
    assert "Retrieved context" in user_msg["content"]


def test_synthesize_includes_recent_history():
    client = _client_with_text("Follow-up answer")
    synth = Synthesizer(client=client, model="claude-sonnet-4-6")
    history = [
        Turn(role="user", content="prev q"),
        Turn(role="assistant", content="prev a"),
    ]
    synth.synthesize(
        question="follow up",
        retrieved=[_ctx(0)],
        canonical_chunks=[],
        history=history,
    )
    msgs = client.messages.create.call_args.kwargs["messages"]
    assert msgs[0]["content"] == "prev q"
    assert msgs[1]["content"] == "prev a"


def test_synthesize_confidence_based_on_citation_count():
    client_high = _client_with_text("A [^1]. B [^2]. C [^3].")
    s_high = Synthesizer(client=client_high, model="claude-sonnet-4-6")
    res = s_high.synthesize(
        question="q",
        retrieved=[_ctx(0), _ctx(1), _ctx(2)],
        canonical_chunks=[],
        history=[],
    )
    assert res.confidence == "high"

    client_low = _client_with_text("no citations here")
    s_low = Synthesizer(client=client_low, model="claude-sonnet-4-6")
    res = s_low.synthesize(question="q", retrieved=[_ctx(0)], canonical_chunks=[], history=[])
    assert res.confidence == "low"


def test_synthesize_web_only_citations_do_not_reach_high_confidence():
    client = _client_with_text("A [^1]. B [^2]. C [^3].")
    synth = Synthesizer(client=client, model="claude-sonnet-4-6")
    result = synth.synthesize(
        question="q",
        retrieved=[_web_ctx(0), _web_ctx(1), _web_ctx(2)],
        canonical_chunks=[],
        history=[],
    )
    assert len(result.citations) == 3
    assert result.confidence != "high"


def test_synthesize_with_adapter_runs_tool_loop():
    from chatqec.conversation.mcp_adapter import MCPAdapter, MCPToolDef

    fake_mcp = MagicMock()
    fake_mcp.list_tools.return_value = [
        MCPToolDef(
            name="stim_simulate",
            description="run",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    fake_mcp.call_tool.return_value = {"ok": True, "summary": "result"}

    client = MagicMock()
    m1 = MagicMock()
    m1.stop_reason = "tool_use"
    b1 = MagicMock()
    b1.type = "tool_use"
    b1.id = "t1"
    b1.name = "stim_simulate"
    b1.input = {}
    m1.content = [b1]
    m1.usage = MagicMock(input_tokens=1, output_tokens=2)
    m2 = MagicMock()
    m2.stop_reason = "end_turn"
    b2 = MagicMock()
    b2.type = "text"
    b2.text = "Answer [^1]"
    m2.content = [b2]
    m2.usage = MagicMock(input_tokens=1, output_tokens=2)
    client.messages.create.side_effect = [m1, m2]

    synth = Synthesizer(
        client=client,
        model="claude-sonnet-4-6",
        mcp_adapter=MCPAdapter(client=fake_mcp),
    )
    result = synth.synthesize(question="q", retrieved=[_ctx(0)], canonical_chunks=[], history=[])
    assert len(result.tool_invocations) == 1
    assert result.tool_invocations[0].name == "stim_simulate"


def test_synthesize_stream_falls_back_to_create_when_client_cannot_stream():
    # FallbackLLMClient exposes create() but not stream(); the stream path must
    # not crash — it should yield the answer via a single create() call.
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text="fallback answer [^1]")]
    messages = MagicMock(spec=["create"])  # no `stream` attribute
    messages.create.return_value = msg
    client.messages = messages

    synth = Synthesizer(client=client, model="m")
    out = "".join(
        synth.synthesize_stream(question="q", retrieved=[_ctx(0)], canonical_chunks=[], history=[])
    )
    assert "fallback answer" in out


def test_synthesize_stream_runs_tool_loop_when_adapter_set():
    from chatqec.conversation.mcp_adapter import MCPAdapter, MCPToolDef

    fake_mcp = MagicMock()
    fake_mcp.list_tools.return_value = [
        MCPToolDef(name="stim_simulate", description="run", input_schema={"type": "object"}),
    ]
    fake_mcp.call_tool.return_value = {"ok": True}

    client = MagicMock()
    m1 = MagicMock(stop_reason="tool_use")
    b1 = MagicMock(type="tool_use", id="t1", input={})
    b1.name = "stim_simulate"  # set after construction (name= is a MagicMock kwarg)
    m1.content = [b1]
    m2 = MagicMock(stop_reason="end_turn")
    m2.content = [MagicMock(type="text", text="Simulated answer [^1]")]
    client.messages.create.side_effect = [m1, m2]

    synth = Synthesizer(client=client, model="m", mcp_adapter=MCPAdapter(client=fake_mcp))
    out = "".join(
        synth.synthesize_stream(question="q", retrieved=[_ctx(0)], canonical_chunks=[], history=[])
    )
    assert "Simulated answer" in out
    fake_mcp.call_tool.assert_called_once()


def test_build_messages_embeds_images_as_blocks():
    from chatqec.api_types import ImageAttachment
    from chatqec.conversation.synthesizer import Synthesizer

    synth = Synthesizer(client=object(), model="m")
    msgs = synth._build_messages(
        question="What is this?",
        retrieved=[],
        canonical_chunks=[],
        history=[],
        images=[ImageAttachment(data=b"x", media_type="image/png")],
    )
    content = msgs[-1]["content"]
    assert isinstance(content, list)
    assert any(b.get("type") == "image" for b in content)
    assert any(b.get("type") == "text" for b in content)


def test_build_messages_plain_string_without_images():
    from chatqec.conversation.synthesizer import Synthesizer

    synth = Synthesizer(client=object(), model="m")
    msgs = synth._build_messages(question="q", retrieved=[], canonical_chunks=[], history=[])
    assert isinstance(msgs[-1]["content"], str)
