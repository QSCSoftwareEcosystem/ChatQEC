"""Tests for the Anthropic-shaped Gemini provider with tool-use translation."""

from types import SimpleNamespace

from chatqec.providers.base import TextBlock, ToolUseBlock
from chatqec.providers.gemini import (
    GeminiLLM,
    _from_gemini_response,
    _sanitize_schema,
    _to_gemini_contents,
)


def test_sanitize_schema_strips_keys_gemini_rejects():
    schema = {
        "type": "object",
        "title": "StimSimulateInput",
        "additionalProperties": False,
        "properties": {
            "circuit": {"type": "string", "title": "Circuit"},
            "shots": {"type": "integer", "minimum": 1, "maximum": 1000000, "default": 1},
        },
        "required": ["circuit"],
    }
    out = _sanitize_schema(schema)
    assert out["type"] == "object"
    assert "title" not in out
    assert "additionalProperties" not in out
    assert out["properties"]["shots"] == {"type": "integer"}
    assert out["required"] == ["circuit"]


def test_sanitize_schema_collapses_optional_any_of_to_its_type():
    # Pydantic's rendering of `rounds: int | None` (threshold_sweep's actual
    # input model) — Gemini's Schema proto has no "anyOf" field and raises
    # ValueError on it rather than ignoring it, so this must not crash.
    schema = {
        "type": "object",
        "properties": {
            "rounds": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Rounds per shot; defaults to distance",
            },
        },
    }
    out = _sanitize_schema(schema)
    assert "anyOf" not in out["properties"]["rounds"]
    assert out["properties"]["rounds"]["type"] == "integer"
    assert out["properties"]["rounds"]["description"] == "Rounds per shot; defaults to distance"
    assert "default" not in out["properties"]["rounds"]


def test_sanitize_schema_resolves_ref_against_defs():
    # glcb_visualize_url's actual shape: a top-level $defs map, with array
    # items pointing at it via $ref. Gemini's Schema proto has neither $ref
    # nor $defs and raises ValueError on either key outright.
    schema = {
        "$defs": {
            "GlcbGate": {
                "type": "object",
                "properties": {"gate": {"type": "string"}, "column": {"type": "integer"}},
                "required": ["gate", "column"],
            }
        },
        "properties": {
            "gates": {"type": "array", "items": {"$ref": "#/$defs/GlcbGate"}},
        },
    }
    out = _sanitize_schema(schema)
    assert "$defs" not in out
    assert out["properties"]["gates"]["items"] == {
        "type": "object",
        "properties": {"gate": {"type": "string"}, "column": {"type": "integer"}},
        "required": ["gate", "column"],
    }


def test_sanitize_schema_degrades_unresolvable_ref_instead_of_crashing():
    schema = {"properties": {"x": {"$ref": "https://external.example/schema.json"}}}
    out = _sanitize_schema(schema)
    assert out["properties"]["x"] == {}


def test_sanitize_schema_drops_true_multi_type_union_instead_of_crashing():
    schema = {
        "properties": {
            "value": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
    }
    out = _sanitize_schema(schema)
    assert "anyOf" not in out["properties"]["value"]
    # No single equivalent type — degrades to untyped rather than raising.
    assert out["properties"]["value"] == {}


def _part(text=None, fn_name=None, fn_args=None):
    fc = SimpleNamespace(name=fn_name, args=fn_args or {}) if fn_name else None
    return SimpleNamespace(text=text, function_call=fc)


def _resp(parts):
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


# --- contents translation (Anthropic messages -> Gemini contents) ---


def test_to_contents_maps_user_text():
    contents = _to_gemini_contents(system=None, messages=[{"role": "user", "content": "hi"}])
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_to_contents_maps_assistant_to_model_role():
    msgs = [{"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]
    contents = _to_gemini_contents(system=None, messages=msgs)
    assert contents[0]["role"] == "model"


def test_to_contents_maps_tool_use_and_tool_result_roundtrip():
    msgs = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "stim_simulate", "input": {"shots": 4}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"ok": true}'}],
        },
    ]
    contents = _to_gemini_contents(system=None, messages=msgs)
    # assistant tool_use -> model function_call
    fc_part = contents[1]["parts"][0]
    assert fc_part["function_call"]["name"] == "stim_simulate"
    assert fc_part["function_call"]["args"] == {"shots": 4}
    # user tool_result -> function_response carrying the originating tool name
    fr_part = contents[2]["parts"][0]
    assert fr_part["function_response"]["name"] == "stim_simulate"
    assert fr_part["function_response"]["response"] == {"ok": True}


# --- response translation (Gemini response -> MessageResponse) ---


def test_from_response_text_only_is_end_turn():
    mr = _from_gemini_response(_resp([_part(text="hello")]))
    assert mr.stop_reason == "end_turn"
    assert mr.content == [TextBlock(text="hello")]


def test_from_response_function_call_is_tool_use():
    mr = _from_gemini_response(_resp([_part(fn_name="stim_simulate", fn_args={"shots": 8})]))
    assert mr.stop_reason == "tool_use"
    block = mr.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "stim_simulate"
    assert block.input == {"shots": 8}
    assert block.id  # a non-empty synthetic id


# --- create() orchestration with an injected fake model ---


def test_create_calls_model_and_maps_response():
    captured = {}

    def fake_factory(*, model, tools, system):
        def generate_content(contents):
            captured["contents"] = contents
            captured["tools"] = tools
            return _resp([_part(text="answer")])

        return SimpleNamespace(generate_content=generate_content)

    llm = GeminiLLM(model_factory=fake_factory)
    resp = llm.messages.create(
        model="gemini-2.5-flash",
        max_tokens=100,
        system="be precise",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "stim_simulate", "description": "run", "input_schema": {"type": "object"}}],
    )
    assert resp.content[0].text == "answer"
    assert captured["contents"] == [{"role": "user", "parts": [{"text": "q"}]}]
