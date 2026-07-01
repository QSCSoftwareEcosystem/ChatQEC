# Design: Image input (multimodal prompts)

Date: 2026-07-01
Status: Approved (brainstorming) — pending implementation plan

## Goal

Let users attach image(s) to a chat prompt and ask questions about them (e.g.
"what does this circuit do?"), the way Claude/GPT do. The image must be *seen*
by the answering model, and it should also help retrieve relevant QEC knowledge
from the corpus.

## Decisions (from brainstorming)

- **Behavior:** "both, auto-decided" — the raw image always goes to the
  answering VLM; an image caption always augments the retrieval query.
  Deterministic (no fragile "is this KB-dependent?" branch).
- **Surfaces:** Streamlit chat **and** CLI (`chatqec ask --image`).
- **Count:** up to **3 images** per message.
- **Approach A** (chosen): full multimodal answer + caption-for-retrieval.
  Rejected: B (caption-only, model never sees image) and C (CLIP image-embedding
  retrieval — YAGNI now).

## Data flow

`ImageAttachment(data: bytes, media_type: str, filename: str = "")` (frozen) is
threaded through the pipeline. When `ask(question, images=[...])` has images:

1. **Caption once** — `conversation/vision.py::describe_image(...)` using
   **Gemini 2.5 Flash**: a short, search-oriented description of the image(s).
2. **Classifier** runs on `question + caption` — on-topic gate still works for an
   image-only prompt (empty text box).
3. **Rewriter** builds retrieval queries from `question + caption` — retrieval
   matches the figure's topic ("image-informed retrieval").
4. **Retrieve** + canonical injection — unchanged.
5. **Synthesizer** receives `images=` and builds **multimodal message content**:
   the retrieved-context+question text block plus the raw image block(s). The VLM
   sees the figure and answers grounded in image + retrieved chunks.

Rule: **image → answerer (always); image-caption → retrieval (always).** The
caption is cheap enough that always-on beats a flaky gate.

## Components

| Component | Change |
|---|---|
| `api_types.py` | Add `ImageAttachment` (frozen). `Answer` unchanged. |
| `conversation/vision.py` (new) | `describe_image(client, model, images) -> str` — Gemini Flash caption for retrieval. |
| `api.py` | `ask/ask_stream` gain `images=[]`; caption → feed classifier + rewriter; pass `images` to synthesizer; trace an `image.caption` step. |
| `conversation/synthesizer.py` | `synthesize/synthesize_stream` gain `images=[]`; `_build_messages` emits the user turn's `content` as a **list of blocks** (text + image) when images present, else the plain string (back-compat). |
| `providers/gemini.py` | `_to_gemini`: image block `{"type":"image",...}` → Gemini `{"inline_data":{mime_type,data}}`. |
| `providers/base.py` / Anthropic | Native `{"type":"image","source":{"type":"base64",...}}` blocks pass through the fallback client unchanged. |
| `providers/huggingface.py` | `_anthropic_to_oai_messages` gains an image branch → OpenAI `{"type":"image_url","image_url":{"url":"data:<mime>;base64,…"}}`; image-bearing calls route to a vision model (Llama 4). |
| `conf/config.yaml` | New `models.hf.vision: meta-llama/Llama-4-Scout-17B-16E-Instruct`. Text roles stay on Llama 3.x. |
| `chatqec_app/pages/1_Chat.py` | `st.chat_input(accept_file="multiple", file_type=["png","jpg","jpeg","webp"])`; validate; build `ImageAttachment`s; show thumbnails in the turn; capability notice when only HF-text is configured. |
| `cli.py` | `chatqec ask --image PATH` (repeatable, ≤3). |
| `ingestion/images.py` (new, shared) | Validate type/count/size + downscale (Pillow) → `list[ImageAttachment]`. Used by Streamlit and CLI so rules can't drift. |

## Provider capability & fallback

Vision requires a multimodal model. With `provider=auto`, the fallback chain
becomes **Anthropic → Gemini → Llama 4 (HF)**, and images ride along on every
leg. Graceful-degrade (drop images + append a "(image not analyzed — vision
unavailable on this backend)" note) is a **last resort**: only when the vision
model is unavailable/errors on images, or a text-only model is explicitly
pinned. The retrieval caption always uses Gemini Flash regardless of the answer
leg.

## Guardrails & privacy

- **Types:** PNG / JPEG / WebP only; reject others with a clear message.
- **Count:** ≤ 3 per message (Streamlit + CLI).
- **Size:** cap ~5 MB per image; downscale to ~1568 px max edge before sending.
- **Abuse/cost:** image `ask` counts against the existing rate limit and cost
  budget; image turns get a higher cost estimate.
- **Privacy:** uploaded images are **transient** — in memory, sent to the model,
  **not persisted** to disk or the vector store. (Distinct from *ingested* book
  figures, which are stored.)

## Testing

- **Unit (no network):** `images.py` validation/downscale; `ImageAttachment`;
  `_build_messages` (string vs block-list); provider translators — `gemini`
  `inline_data` and HF `image_url` shapes — with mock clients.
- **api flow:** `ask(question, images=[...])` with a fake client asserts the
  caption step runs, the retrieval query includes the caption, and the
  synthesizer receives images.
- **Degrade path:** text-only backend drops images and emits the notice.
- **Manual/integration:** one real Streamlit round-trip with a QEC figure
  (Gemini) — verified by hand, not in CI (as with Marker/embeddings).

## Out of scope (YAGNI)

- CLIP / image-embedding retrieval (approach C).
- Persisting user-uploaded images.
- Image *generation*.
- Video / PDF-as-image upload (PDFs already have the ingestion path).

## Open risks

- HF serverless may not reliably serve a Llama-4 vision model or accept
  `image_url` parts; the degrade path is the safety net.
- Llama 4 vision quality on dense scientific diagrams is below Gemini/Claude —
  acceptable as the third fallback.
