"""Gemini provider translation tests."""


def test_to_gemini_contents_maps_image_block():
    from chatqec.providers.gemini import _to_gemini_contents

    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                },
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    contents = _to_gemini_contents(system=None, messages=msgs)
    parts = contents[0]["parts"]
    assert {"inline_data": {"mime_type": "image/png", "data": "QUJD"}} in parts
    assert {"text": "what is this?"} in parts
