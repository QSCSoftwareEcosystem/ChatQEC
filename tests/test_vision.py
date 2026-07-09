from unittest.mock import MagicMock

from chatqec.api_types import ImageAttachment
from chatqec.conversation.vision import describe_image

IMG = ImageAttachment(data=b"x", media_type="image/png")


def test_returns_empty_without_images():
    client = MagicMock()
    assert describe_image(client=client, model="m", images=[]) == ""
    client.messages.create.assert_not_called()


def test_captions_via_client():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text="A surface-code lattice.")]
    client.messages.create.return_value = msg
    out = describe_image(client=client, model="gemini-2.5-flash", images=[IMG])
    assert out == "A surface-code lattice."
    kwargs = client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)
