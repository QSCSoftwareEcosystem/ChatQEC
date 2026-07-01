"""HuggingFace provider message-translation tests."""

from chatqec.providers.huggingface import _anthropic_to_oai_messages, has_image_content


def test_oai_messages_include_image_url():
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                },
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    out = _anthropic_to_oai_messages(system=None, messages=msgs)
    parts = out[0]["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}} in parts
    assert {"type": "text", "text": "hi"} in parts


def test_text_only_stays_string():
    out = _anthropic_to_oai_messages(system="s", messages=[{"role": "user", "content": "plain"}])
    assert out[-1]["content"] == "plain"


def test_has_image_content():
    assert has_image_content([{"role": "user", "content": [{"type": "image"}]}]) is True
    assert has_image_content([{"role": "user", "content": "plain"}]) is False
