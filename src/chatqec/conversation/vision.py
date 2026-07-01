"""Image captioning for retrieval (Gemini Flash by default).

Turns attached images into a short, search-oriented description that is folded
into the retrieval query, so KB context matches the figure.
"""

from __future__ import annotations

from typing import Any

from chatqec.api_types import ImageAttachment
from chatqec.conversation.image_blocks import anthropic_image_block

_SYS = (
    "You describe a scientific figure for search. In 1-3 sentences, name the "
    "concepts, objects, and QEC terms it depicts (codes, circuits, lattices, "
    "decoders, plots). No preamble."
)


def describe_image(*, client: Any, model: str, images: list[ImageAttachment]) -> str:
    """Return a short search-oriented description of ``images`` (``""`` if none)."""
    if not images:
        return ""
    content: list[dict[str, Any]] = [anthropic_image_block(i) for i in images]
    content.append({"type": "text", "text": "Describe this figure for search."})
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        system=_SYS,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
