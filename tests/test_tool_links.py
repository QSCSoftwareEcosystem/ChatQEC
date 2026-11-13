"""Long tool links: placeholder for the model, real links for the user."""

from chatqec.api_types import ToolCall
from chatqec.conversation.tool_links import PLACEHOLDER, links_from_tool_calls, slim_long_urls
from chatqec.conversation.tool_loop import _serialize_tool_result

LONG = "https://bloqade.quera.com/studio/qec/#1" + "A" * 2000
SHORT = "https://bloqade.quera.com/studio/gemini/#circuit={version:1.1}"


def test_long_urls_become_placeholders_nested_short_ones_stay():
    out = slim_long_urls({"url": LONG, "examples": [{"url": LONG, "title": "x"}], "s": SHORT})
    assert out == {"url": PLACEHOLDER, "examples": [{"url": PLACEHOLDER, "title": "x"}], "s": SHORT}


def test_model_never_sees_the_long_link():
    assert LONG not in _serialize_tool_result({"ok": True, "url": LONG})


def test_links_collected_from_recorded_outputs_in_order_without_duplicates():
    calls = [
        ToolCall(name="qec_studio_link", input={}, output=str({"ok": True, "url": LONG})),
        ToolCall(
            name="qec_studio_examples",
            input={},
            output=str(
                {"ok": True, "examples": [{"title": "MSD", "url": LONG}, {"title": "B", "url": ""}]}
            ),
        ),
        ToolCall(name="glcb_visualize_url", input={}, output=str({"ok": True, "url": SHORT})),
        ToolCall(name="stim_simulate", input={}, output="not a dict"),
    ]
    assert links_from_tool_calls(calls) == [
        ("qec_studio_link", LONG),
        ("glcb_visualize_url", SHORT),
    ]
