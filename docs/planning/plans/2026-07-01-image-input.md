# Image Input (Multimodal Prompts) Implementation Plan

**Goal:** Let users attach up to 3 images to a chat prompt (Streamlit + CLI) and ask questions about them; the answering VLM sees the image and a Gemini-Flash caption augments retrieval.

**Architecture:** An `ImageAttachment` value object threads through `ChatQEC.ask/ask_stream` → synthesizer (multimodal message blocks) → provider translators. Images always reach the answerer; a caption always augments retrieval. Fallback chain Anthropic→Gemini→Llama-4(HF) all carry images; graceful-degrade is a last resort.

**Tech Stack:** Python 3.11, Click, Streamlit 1.58, Pillow, google-generativeai (Gemini), anthropic, huggingface_hub; pytest/ruff/mypy.

## Global Constraints

- Type hints on all functions; ruff + mypy clean; pytest green (`uv run ruff check .`, `uv run mypy src/...`, `uv run pytest`).
- Frozen dataclasses for value objects (match `api_types.py`).
- No network in unit tests — mock clients (match existing test style).
- Image guardrails (verbatim): PNG/JPEG/WebP only; ≤ 3 images/message; ≤ 5 MB each; downscale to ≤ 1568 px max edge.
- Uploaded images are transient — never written to disk or the vector store.
- Retrieval caption model: `gemini-2.5-flash`. HF vision model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`.
- Back-compat: message `content` stays a plain `str` when no images; becomes a block list only when images are present.

---

### Task 1: `ImageAttachment` value object + validation/downscale helper

**Files:**
- Modify: `src/chatqec/api_types.py`
- Create: `src/chatqec/ingestion/images.py`
- Test: `tests/test_images.py`

**Interfaces:**
- Produces: `ImageAttachment(data: bytes, media_type: str, filename: str = "")` (frozen);
  `prepare_images(raw: list[tuple[bytes, str, str]], *, max_images: int = 3, max_bytes: int = 5_000_000, max_edge: int = 1568) -> list[ImageAttachment]` — validates type/count/size, downscales, returns attachments; raises `ImageError` on bad type/too many.
  `class ImageError(ValueError)`.
  `ALLOWED_MEDIA_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_images.py
import io
import pytest
from PIL import Image
from chatqec.api_types import ImageAttachment
from chatqec.ingestion.images import ImageError, prepare_images, ALLOWED_MEDIA_TYPES


def _png(w=10, h=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_prepare_returns_attachments():
    out = prepare_images([(_png(), "image/png", "a.png")])
    assert len(out) == 1
    assert isinstance(out[0], ImageAttachment)
    assert out[0].media_type == "image/png"
    assert out[0].filename == "a.png"
    assert out[0].data  # non-empty


def test_rejects_disallowed_type():
    with pytest.raises(ImageError):
        prepare_images([(b"gif89a", "image/gif", "x.gif")])


def test_rejects_too_many():
    imgs = [(_png(), "image/png", f"{i}.png") for i in range(4)]
    with pytest.raises(ImageError):
        prepare_images(imgs, max_images=3)


def test_downscales_large_image():
    big = _png(4000, 100)  # 4000px wide → must shrink to <=1568
    out = prepare_images([(big, "image/png", "big.png")], max_edge=1568)
    w, h = Image.open(io.BytesIO(out[0].data)).size
    assert max(w, h) <= 1568


def test_png_media_type_in_allowed():
    assert "image/png" in ALLOWED_MEDIA_TYPES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_images.py -q`
Expected: FAIL (`ImportError`: cannot import `ImageAttachment` / `prepare_images`).

- [ ] **Step 3: Add `ImageAttachment` to `api_types.py`**

Append after `StreamChunk`:

```python
@dataclass(frozen=True)
class ImageAttachment:
    data: bytes
    media_type: str
    filename: str = ""
```

- [ ] **Step 4: Create `src/chatqec/ingestion/images.py`**

```python
"""Validate + normalize user-uploaded images into ImageAttachment objects.

Shared by the Streamlit chat and the CLI so guardrails can't drift. Images are
transient — this never writes to disk or the vector store.
"""

from __future__ import annotations

import io

from PIL import Image

from chatqec.api_types import ImageAttachment

ALLOWED_MEDIA_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}
_MEDIA_TO_PIL = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


class ImageError(ValueError):
    """Raised when an uploaded image violates the guardrails."""


def _downscale(data: bytes, media_type: str, max_edge: int) -> bytes:
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_edge:
        return data
    img.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, format=_MEDIA_TO_PIL[media_type])
    return buf.getvalue()


def prepare_images(
    raw: list[tuple[bytes, str, str]],
    *,
    max_images: int = 3,
    max_bytes: int = 5_000_000,
    max_edge: int = 1568,
) -> list[ImageAttachment]:
    """Validate (type/count/size) and downscale raw ``(data, media_type, name)``."""
    if len(raw) > max_images:
        raise ImageError(f"at most {max_images} images per message (got {len(raw)})")
    out: list[ImageAttachment] = []
    for data, media_type, name in raw:
        if media_type not in ALLOWED_MEDIA_TYPES:
            raise ImageError(f"unsupported image type: {media_type!r}")
        if len(data) > max_bytes:
            raise ImageError(f"{name or 'image'} exceeds {max_bytes // 1_000_000} MB")
        out.append(
            ImageAttachment(
                data=_downscale(data, media_type, max_edge),
                media_type=media_type,
                filename=name,
            )
        )
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_images.py -q` → Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/api_types.py src/chatqec/ingestion/images.py tests/test_images.py
git commit -m "feat(images): ImageAttachment + validation/downscale helper"
```

---

### Task 2: Image content-block builders (Anthropic / Gemini / OpenAI shapes)

**Files:**
- Create: `src/chatqec/conversation/image_blocks.py`
- Test: `tests/test_image_blocks.py`

**Interfaces:**
- Consumes: `ImageAttachment` (Task 1).
- Produces:
  `anthropic_image_block(img: ImageAttachment) -> dict` → `{"type":"image","source":{"type":"base64","media_type":..,"data":<b64 str>}}`
  `openai_image_part(img: ImageAttachment) -> dict` → `{"type":"image_url","image_url":{"url":"data:<mt>;base64,<b64>"}}`
  `gemini_image_part(img: ImageAttachment) -> dict` → `{"inline_data":{"mime_type":..,"data":<b64 str>}}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_blocks.py
import base64
from chatqec.api_types import ImageAttachment
from chatqec.conversation.image_blocks import (
    anthropic_image_block, openai_image_part, gemini_image_part,
)

IMG = ImageAttachment(data=b"\x89PNG\r\n", media_type="image/png", filename="a.png")
B64 = base64.b64encode(IMG.data).decode()


def test_anthropic_block():
    b = anthropic_image_block(IMG)
    assert b["type"] == "image"
    assert b["source"] == {"type": "base64", "media_type": "image/png", "data": B64}


def test_openai_part():
    p = openai_image_part(IMG)
    assert p["type"] == "image_url"
    assert p["image_url"]["url"] == f"data:image/png;base64,{B64}"


def test_gemini_part():
    p = gemini_image_part(IMG)
    assert p["inline_data"] == {"mime_type": "image/png", "data": B64}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_image_blocks.py -q` → Expected: FAIL (ImportError).

- [ ] **Step 3: Create `src/chatqec/conversation/image_blocks.py`**

```python
"""Convert ImageAttachment into each provider's image content shape."""

from __future__ import annotations

import base64
from typing import Any

from chatqec.api_types import ImageAttachment


def _b64(img: ImageAttachment) -> str:
    return base64.b64encode(img.data).decode()


def anthropic_image_block(img: ImageAttachment) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": img.media_type, "data": _b64(img)},
    }


def openai_image_part(img: ImageAttachment) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{img.media_type};base64,{_b64(img)}"},
    }


def gemini_image_part(img: ImageAttachment) -> dict[str, Any]:
    return {"inline_data": {"mime_type": img.media_type, "data": _b64(img)}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_image_blocks.py -q` → Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/conversation/image_blocks.py tests/test_image_blocks.py
git commit -m "feat(images): per-provider image content-block builders"
```

---

### Task 3: Vision caption for retrieval (`conversation/vision.py`)

**Files:**
- Create: `src/chatqec/conversation/vision.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: `ImageAttachment` (Task 1), `anthropic_image_block` (Task 2).
- Produces: `describe_image(*, client: Any, model: str, images: list[ImageAttachment]) -> str`
  — sends images + a fixed instruction via `client.messages.create(...)`, returns a
  short plain-text description. Returns `""` for empty `images`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vision.py
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
    # image block is included in the user message content
    kwargs = client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vision.py -q` → Expected: FAIL (ImportError).

- [ ] **Step 3: Create `src/chatqec/conversation/vision.py`**

```python
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
    if not images:
        return ""
    content: list[dict[str, Any]] = [anthropic_image_block(i) for i in images]
    content.append({"type": "text", "text": "Describe this figure for search."})
    msg = client.messages.create(
        model=model, max_tokens=200, system=_SYS, messages=[{"role": "user", "content": content}]
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vision.py -q` → Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/conversation/vision.py tests/test_vision.py
git commit -m "feat(images): Gemini-Flash image captioning for retrieval"
```

---

### Task 4: Synthesizer multimodal message content

**Files:**
- Modify: `src/chatqec/conversation/synthesizer.py`
- Test: `tests/test_conversation_synthesizer.py`

**Interfaces:**
- Consumes: `ImageAttachment` (Task 1), `anthropic_image_block` (Task 2).
- Produces: `Synthesizer.synthesize(..., images: list[ImageAttachment] | None = None)` and
  `synthesize_stream(..., images=None)`; `_build_messages(..., images=None)` returns the
  final user turn with `content` as a **block list** (text block + image blocks) when
  `images`, else the existing plain string.

- [ ] **Step 1: Write the failing test** (append to `tests/test_conversation_synthesizer.py`)

```python
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
    msgs = synth._build_messages(
        question="q", retrieved=[], canonical_chunks=[], history=[]
    )
    assert isinstance(msgs[-1]["content"], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conversation_synthesizer.py -k build_messages -q`
Expected: FAIL (`_build_messages` has no `images` kwarg / returns str).

- [ ] **Step 3: Modify `_build_messages` and the two public methods**

Add the import near the top:

```python
from chatqec.api_types import Citation, ImageAttachment
from chatqec.conversation.image_blocks import anthropic_image_block
```

Replace `_build_messages` with:

```python
    def _build_messages(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        canonical_chunks: list[RetrievedChunk],
        history: list[Turn],
        images: list[ImageAttachment] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": t.role, "content": t.content} for t in history
        ]
        context = assemble_context_block(retrieved, canonical_chunks=canonical_chunks)
        text = f"{context}\n\nQuestion: {question}"
        if images:
            content: list[dict[str, Any]] = [anthropic_image_block(i) for i in images]
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": text})
        return messages
```

Add `images: list[ImageAttachment] | None = None` to the signatures of
`synthesize` and `synthesize_stream`, and pass `images=images` into each
`self._build_messages(...)` call inside them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conversation_synthesizer.py -q` → Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/conversation/synthesizer.py tests/test_conversation_synthesizer.py
git commit -m "feat(images): synthesizer builds multimodal message content"
```

---

### Task 5: Gemini provider image branch

**Files:**
- Modify: `src/chatqec/providers/gemini.py` (`_to_gemini_contents`, ~lines 60-84)
- Test: `tests/test_provider_gemini.py`

**Interfaces:**
- Consumes: `gemini_image_part` (Task 2).
- Produces: `_to_gemini_contents` maps an `{"type":"image",...}` (Anthropic-shape) block
  to a Gemini `{"inline_data": {...}}` part.

- [ ] **Step 1: Write the failing test** (append to `tests/test_provider_gemini.py`; create if absent with `from chatqec.providers.gemini import _to_gemini_contents`)

```python
def test_to_gemini_contents_maps_image_block():
    from chatqec.providers.gemini import _to_gemini_contents

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    contents = _to_gemini_contents(system=None, messages=msgs)
    parts = contents[0]["parts"]
    assert {"inline_data": {"mime_type": "image/png", "data": "QUJD"}} in parts
    assert {"text": "what is this?"} in parts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_gemini.py -k image -q`
Expected: FAIL (image block ignored — not in parts).

- [ ] **Step 3: Add the image branch in `_to_gemini_contents`**

Inside the `for b in content:` loop, add before the `tool_use` branch:

```python
                if btype == "image":
                    src = b.get("source", {})
                    parts.append(
                        {"inline_data": {"mime_type": src.get("media_type", ""), "data": src.get("data", "")}}
                    )
                elif btype == "text":
```

(Change the existing `if btype == "text":` to `elif btype == "text":` so it chains.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_gemini.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/providers/gemini.py tests/test_provider_gemini.py
git commit -m "feat(images): gemini provider translates image blocks to inline_data"
```

---

### Task 6: HuggingFace provider image branch (Llama 4 vision)

**Files:**
- Modify: `src/chatqec/providers/huggingface.py` (`_anthropic_to_oai_messages`)
- Test: `tests/test_provider_huggingface.py`

**Interfaces:**
- Consumes: `openai_image_part` (Task 2).
- Produces: `_anthropic_to_oai_messages` emits OpenAI multimodal content (list of
  `{"type":"text",...}` / `{"type":"image_url",...}` parts) when a message has image
  blocks; plain string otherwise. Return type widens to `list[dict[str, Any]]`.
- `has_image_content(messages: list[dict[str, Any]]) -> bool` helper (used by Task 7).

- [ ] **Step 1: Write the failing test** (append to `tests/test_provider_huggingface.py`; create if absent)

```python
def test_oai_messages_include_image_url():
    from chatqec.providers.huggingface import _anthropic_to_oai_messages

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    out = _anthropic_to_oai_messages(system=None, messages=msgs)
    parts = out[0]["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}} in parts
    assert {"type": "text", "text": "hi"} in parts


def test_has_image_content():
    from chatqec.providers.huggingface import has_image_content

    assert has_image_content([{"role": "user", "content": [{"type": "image"}]}]) is True
    assert has_image_content([{"role": "user", "content": "plain"}]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_huggingface.py -k "image" -q` → Expected: FAIL.

- [ ] **Step 3: Update `_anthropic_to_oai_messages` + add `has_image_content`**

Add import: `from chatqec.conversation.image_blocks import openai_image_part` and
`from chatqec.api_types import ImageAttachment`. Replace the `else` branch of the
content loop:

```python
        else:
            parts: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") == "text":
                    parts.append({"type": "text", "text": str(b.get("text", ""))})
                elif b.get("type") == "image":
                    src = b.get("source", {})
                    img = ImageAttachment(
                        data=b"", media_type=str(src.get("media_type", "image/png"))
                    )
                    part = openai_image_part(img)
                    part["image_url"]["url"] = f"data:{img.media_type};base64,{src.get('data','')}"
                    parts.append(part)
            out.append({"role": role, "content": parts})
```

Change the function return annotation to `list[dict[str, Any]]`. Add at module scope:

```python
def has_image_content(messages: list[dict[str, Any]]) -> bool:
    """True if any message carries an image content block."""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(
            isinstance(b, dict) and b.get("type") == "image" for b in c
        ):
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_huggingface.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/providers/huggingface.py tests/test_provider_huggingface.py
git commit -m "feat(images): HF provider emits OpenAI image_url parts (Llama 4 vision)"
```

---

### Task 7: Fallback routes image requests to a vision model

**Files:**
- Modify: `src/chatqec/providers/fallback.py` (`Backend`, `_FallbackMessages.create`)
- Test: `tests/test_provider_fallback.py`

**Interfaces:**
- Consumes: `has_image_content` (Task 6).
- Produces: `Backend` gains `vision_model: str | None = None`. In `create`, when the
  request has image content and `be.vision_model` is set, that backend uses
  `vision_model` instead of the normal model resolution.

- [ ] **Step 1: Write the failing test** (append to `tests/test_provider_fallback.py`)

```python
def test_image_request_uses_vision_model():
    from unittest.mock import MagicMock
    from chatqec.providers.fallback import Backend, FallbackLLMClient
    from chatqec.providers.base import MessageResponse, TextBlock

    inner = MagicMock()
    inner.messages.create.return_value = MessageResponse(content=[TextBlock(text="ok")])
    be = Backend(name="hf", client=inner, default_model="text-model", vision_model="vision-model")
    client = FallbackLLMClient(backends=[be], verbose=False)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
    )
    assert inner.messages.create.call_args.kwargs["model"] == "vision-model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_fallback.py -k vision -q`
Expected: FAIL (`Backend` has no `vision_model` / model is `text-model`).

- [ ] **Step 3: Add `vision_model` to `Backend` and use it in `create`**

In the `Backend` dataclass add: `vision_model: str | None = None`.

In `_FallbackMessages.create`, add near the top:
`from chatqec.providers.huggingface import has_image_content` (local import to
avoid a cycle) and compute `image_req = has_image_content(messages or [])`.
Change the model resolution line:

```python
            if image_req and be.vision_model:
                eff_model = be.vision_model
            else:
                eff_model = be.model_remap.get(model) or be.default_model or model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_fallback.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/providers/fallback.py tests/test_provider_fallback.py
git commit -m "feat(images): fallback routes image requests to a backend vision model"
```

---

### Task 8: Wire images through `ChatQEC.ask` / `ask_stream`

**Files:**
- Modify: `src/chatqec/api.py`
- Test: `tests/test_api.py` (create if absent)

**Interfaces:**
- Consumes: `describe_image` (Task 3), synthesizer `images=` (Task 4), `ImageAttachment`.
- Produces: `ChatQEC.ask(question, images: list[ImageAttachment] | None = None) -> Answer`
  and `ask_stream(question, images=None)`. Caption (via `describe_image` using the
  rewriter's client+model) is appended to the classifier + rewriter inputs and the
  retrieval queries; `images` is passed to the synthesizer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
from unittest.mock import MagicMock
from chatqec.api import ChatQEC
from chatqec.api_types import ImageAttachment


def _bot():
    classifier = MagicMock(); classifier.classify.return_value = MagicMock(on_topic=True, refusal_message="")
    rewriter = MagicMock()
    rewriter.rewrite.return_value = MagicMock(rewritten="q", variants=[], is_followup=False)
    rewriter.client = MagicMock(); rewriter.model = "gemini-2.5-flash"
    retriever = MagicMock()
    retriever.retrieve.return_value = MagicMock(chunks=[], matched_topics=[], debug={})
    synth = MagicMock()
    synth.synthesize.return_value = MagicMock(
        text="ans", citations=[], confidence="low", tool_invocations=[], input_tokens=0, output_tokens=0
    )
    canonical = MagicMock(); canonical.pages = []
    return ChatQEC(classifier=classifier, rewriter=rewriter, retriever=retriever,
                   synthesizer=synth, canonical_index=canonical), synth


def test_ask_passes_images_to_synthesizer(monkeypatch):
    bot, synth = _bot()
    monkeypatch.setattr("chatqec.api.describe_image", lambda **k: "a lattice figure")
    img = ImageAttachment(data=b"x", media_type="image/png")
    bot.ask("what is this?", images=[img])
    assert synth.synthesize.call_args.kwargs["images"] == [img]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -q` → Expected: FAIL (`ask` has no `images` kwarg).

- [ ] **Step 3: Modify `api.py`**

Add imports:

```python
from chatqec.api_types import Answer, Citation, ImageAttachment, StreamChunk, ToolCall
from chatqec.conversation.vision import describe_image
```

In `ask` and `ask_stream`, change the signature to accept
`images: list[ImageAttachment] | None = None`, then at the top of each body:

```python
        images = images or []
        caption = describe_image(
            client=self.rewriter.client, model=self.rewriter.model, images=images
        ) if images else ""
        trace("image.caption", n_images=len(images), caption=caption)
        classify_text = f"{question}\n{caption}".strip() if caption else question
```

Use `classify_text` in `self.classifier.classify(...)`. After the rewrite, when
`caption`, append it to `queries`: `queries = [rewrite.rewritten, *rewrite.variants]`
then `if caption: queries.append(caption)`. Pass `images=images` to both
`self.synthesizer.synthesize(...)` and `self.synthesizer.synthesize_stream(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py tests/ -q -k "api or synthes"` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/api.py tests/test_api.py
git commit -m "feat(images): thread images through ChatQEC.ask (+caption for retrieval)"
```

---

### Task 9: CLI `ask --image` + config vision model

**Files:**
- Modify: `src/chatqec/cli.py` (`ask` command, `_build_llm_client` HF backend), `conf/config.yaml`
- Test: `tests/test_cli_ask_image.py`

**Interfaces:**
- Consumes: `prepare_images` (Task 1), `ChatQEC.ask(images=)` (Task 8), `Backend.vision_model` (Task 7).
- Produces: `chatqec ask --image PATH` (repeatable, ≤3); HF `Backend` built with
  `vision_model` from `models.hf.vision`.

- [ ] **Step 1: Write the failing test** (helper `_read_image_files` is the unit under test)

```python
# tests/test_cli_ask_image.py
import io
from pathlib import Path
from PIL import Image
from chatqec.cli import _read_image_files


def test_read_image_files(tmp_path: Path):
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p)
    imgs = _read_image_files((str(p),))
    assert len(imgs) == 1
    assert imgs[0].media_type == "image/png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_ask_image.py -q` → Expected: FAIL (ImportError).

- [ ] **Step 3: Add `_read_image_files` + `--image` option + config**

In `cli.py` add:

```python
def _read_image_files(paths: tuple[str, ...]) -> list:  # -> list[ImageAttachment]
    from chatqec.ingestion.images import prepare_images

    _ext_to_media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    raw = []
    for p in paths:
        path = Path(p)
        media = _ext_to_media.get(path.suffix.lower(), "image/png")
        raw.append((path.read_bytes(), media, path.name))
    return prepare_images(raw)
```

Add to the `ask` command: `@click.option("--image", "image_paths", multiple=True, type=click.Path(exists=True, dir_okay=False), help="Attach an image (repeatable, ≤3).")`, add `image_paths: tuple[str, ...]` to the signature, and call
`answer = bot.ask(" ".join(question), images=_read_image_files(image_paths))`.

In `_build_llm_client`, when building the HF `Backend` in the `auto` branch, add
`vision_model=str(_HF_VISION_MODEL)` where `_HF_VISION_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"` (module constant). In `conf/config.yaml` under `models.hf` add:
`vision: meta-llama/Llama-4-Scout-17B-16E-Instruct`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_ask_image.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/cli.py conf/config.yaml tests/test_cli_ask_image.py
git commit -m "feat(images): chatqec ask --image + HF vision model config"
```

---

### Task 10: Streamlit chat image attach + render

**Files:**
- Modify: `chatqec_app/pages/1_Chat.py`
- Test: manual (Streamlit round-trip — no unit test; matches app convention)

**Interfaces:**
- Consumes: `prepare_images` (Task 1), `ChatQEC.ask_stream/ask(images=)` (Task 8),
  `ImageError` (Task 1).

- [ ] **Step 1: Enable file attach on the chat input**

Replace `question = st.chat_input("Ask a QEC question…")` with:

```python
submission = st.chat_input(
    "Ask a QEC question…",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)
question = submission.text if submission else None
uploaded = submission.files if submission else []
```

- [ ] **Step 2: Build attachments (guardrailed) and show them**

After the rate-limit/budget checks, before calling the bot:

```python
    from chatqec.ingestion.images import ImageError, prepare_images

    images = []
    if uploaded:
        try:
            images = prepare_images([(f.getvalue(), f.type, f.name) for f in uploaded])
        except ImageError as e:
            st.error(str(e))
            st.stop()
    with st.chat_message("user"):
        st.markdown(question or "")
        for im in images:
            st.image(im.data, width=240)
```

- [ ] **Step 3: Pass images into the calls**

Change `bot.ask_stream(question)` → `bot.ask_stream(question or "", images=images)` and
the follow-up `bot.ask(question)` → `bot.ask(question or "", images=images)`.

- [ ] **Step 4: Image cost surcharge (coarse)**

After the answer, if `images`:

```python
    if images:
        get_cost_budget().charge(amount_usd=0.01 * len(images), now=utc_now())
```

- [ ] **Step 5: Manual verification**

Run: `uv run streamlit run chatqec_app/streamlit_app.py`
- Attach a QEC figure (PNG), ask "what does this show?" → answer references the figure.
- Attach 4 images → see the "at most 3" error.

- [ ] **Step 6: Commit**

```bash
git add chatqec_app/pages/1_Chat.py
git commit -m "feat(images): Streamlit chat image attach + render"
```

---

### Task 11: Full verification

- [ ] **Step 1:** `uv run ruff check .` → All checks passed.
- [ ] **Step 2:** `uv run mypy src/chatqec` → Success.
- [ ] **Step 3:** `uv run pytest -q` → all pass.
- [ ] **Step 4:** Update `CHANGELOG.md` `[Unreleased] → Added`: "Image input (multimodal prompts): attach up to 3 images in the Streamlit chat or `chatqec ask --image`; the VLM sees the figure and a Gemini-Flash caption augments retrieval. Fallback Anthropic→Gemini→Llama-4 all carry images."
- [ ] **Step 5:** Commit: `git commit -am "docs(changelog): record image input feature"`

---

## Self-Review

- **Spec coverage:** ImageAttachment/validation (T1), caption-for-retrieval (T3, T8), multimodal answer (T4), Gemini branch (T5), HF/Llama-4 branch + vision routing (T6, T7, T9 config), Streamlit + CLI surfaces (T9, T10), guardrails (T1, T10), transient/no-persist (T1 docstring, T10 uses in-memory bytes), degrade path (fallback chain in T7 + existing `_is_recoverable`), testing (unit per task + manual T10). All spec sections mapped.
- **Placeholder scan:** none — every code step has complete code.
- **Type consistency:** `ImageAttachment(data,media_type,filename)`, `prepare_images`, `describe_image(*, client, model, images)`, `anthropic_image_block/openai_image_part/gemini_image_part`, `has_image_content`, `Backend.vision_model` are used with identical signatures across tasks.
- **Note:** Task 6's `openai_image_part` is called then its URL overwritten with the base64 from the source block (the block carries base64, not an `ImageAttachment`) — intentional, keeps one URL format in Task 2.
