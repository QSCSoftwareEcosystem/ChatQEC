# Figure Vision Ingestion (Level 3) Implementation Plan

**Goal:** Persist book/paper figures during ingestion, describe each with a local vision model so figures are semantically searchable, and surface the actual image to the user at answer time.

**Architecture:** Marker already emits figures as `![](name.jpeg)` next to its markdown. We (1) copy those images into a durable `data/figures/<source_id>/` folder and fix the links (DONE — `persist_marker_images`), (2) run a local **Llama-3.2-11B-Vision-Instruct** describer over each real figure and inject a marked description block into the markdown before chunking, (3) record each chunk's figure paths in `Chunk.metadata`, and (4) at answer time load those images from retrieved chunks and pass them to the multimodal synthesizer + render them (Streamlit) / print their paths (CLI). The describer sits behind a `FigureDescriber` protocol so the model is swappable and tests use a fake.

**Tech Stack:** Python 3.11, `transformers` (`MllamaForConditionalGeneration` + `AutoProcessor`), `torch` (CUDA), `Pillow`, Click CLI, Streamlit, pytest.

## Global Constraints

- **US-approved open-weights only.** Vision model is `meta-llama/Llama-3.2-11B-Vision-Instruct` (Meta). No China-origin models (matches `conf/config.yaml` reranker policy).
- **Local inference, zero API cost.** The describer runs on local GPUs via `transformers`; never route figure description through a paid API.
- **Image persistence must never break text ingestion.** Any describer/copy failure is caught and logged; ingestion falls back to text-only for that figure.
- **File size:** keep new modules 200–400 lines (repo `coding-style.md`).
- **`data/figures/` is a build artifact** — gitignore it (like `data/chunks/`). The registry + chunk text are the tracked source of truth.
- **Descriptions are model-generated** — always wrap them in a `> **Figure (auto-described):** …` blockquote so they are never mistaken for source prose or cited as such.
- **TDD.** Every new function gets a failing test first. The model impl itself is `# pragma: no cover` (GPU + weights); everything around it is tested with a fake describer.

---

## Increment A — Ingest, persist, describe, index (Tasks 1–5)

Produces: figures saved to disk, described, and searchable; chunks carry figure paths. Shippable on its own.

### Task 1: Gitignore `data/figures/` + commit the persistence foundation

**Files:**
- Modify: `.gitignore`
- (already in working tree) `src/chatqec/ingestion/fetchers/pdf_marker.py` (`persist_marker_images`, `run_marker_on_pdf` images_dir arg)
- (already in working tree) `tests/test_pdf_marker.py` (4 new tests)

**Interfaces:**
- Produces: `persist_marker_images(md_text: str, src_dir: Path, dest_dir: Path) -> str` and `run_marker_on_pdf(pdf_path, *, timeout=1200, page_range=None, images_dir: Path | None = None) -> str`.

- [ ] **Step 1:** Add `data/figures/` under the `data/chunks/` line in `.gitignore`.

```gitignore
data/chunks/
data/figures/
```

- [ ] **Step 2:** Run the already-written persistence tests to confirm green.

Run: `.venv/bin/python -m pytest tests/test_pdf_marker.py -q --no-cov`
Expected: PASS (10 passed)

- [ ] **Step 3:** Commit.

```bash
git add .gitignore src/chatqec/ingestion/fetchers/pdf_marker.py tests/test_pdf_marker.py
git commit -m "feat(ingest): persist Marker figures to data/figures and fix links"
```

---

### Task 2: Decorative-image filter

Marker extracts page furniture (logos, rules, cover art like `_page_0_Picture_0.jpeg`). Skip them so we don't waste vision calls or pollute embeddings.

**Files:**
- Create: `src/chatqec/ingestion/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Produces: `is_probable_figure(path: Path, *, min_bytes: int = 3000, min_edge: int = 96) -> bool` — True if the image is large enough (bytes AND min dimension) to be a real figure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures.py
import io
from PIL import Image
from chatqec.ingestion.figures import is_probable_figure


def _png(tmp_path, name, size):
    p = tmp_path / name
    Image.new("RGB", size, (200, 200, 200)).save(p, format="PNG")
    return p


def test_rejects_tiny_dimension_image(tmp_path):
    assert is_probable_figure(_png(tmp_path, "icon.png", (32, 32))) is False


def test_accepts_reasonable_figure(tmp_path):
    assert is_probable_figure(_png(tmp_path, "fig.png", (640, 480))) is True


def test_rejects_missing_file(tmp_path):
    assert is_probable_figure(tmp_path / "nope.png") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_figures.py -q --no-cov`
Expected: FAIL (`ModuleNotFoundError: chatqec.ingestion.figures`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/chatqec/ingestion/figures.py
"""Heuristics for figure images extracted from PDFs during ingestion."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def is_probable_figure(path: Path, *, min_bytes: int = 3000, min_edge: int = 96) -> bool:
    """True if ``path`` is a real figure (not decorative page furniture).

    Cheap two-part gate: the file must be non-trivially large on disk AND its
    smaller dimension must exceed ``min_edge`` px. Missing/unreadable files are
    rejected rather than raising.
    """
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        with Image.open(path) as im:
            w, h = im.size
    except OSError:
        return False
    return min(w, h) >= min_edge
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_figures.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/figures.py tests/test_figures.py
git commit -m "feat(ingest): decorative-image filter for figure extraction"
```

---

### Task 3: `FigureDescriber` protocol + `annotate_figures`

Inject a marked description block after each real figure link. The describer is injected (a fake in tests); the real model comes in Task 4.

**Files:**
- Modify: `src/chatqec/ingestion/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `is_probable_figure` (Task 2); image links produced by `persist_marker_images` (Task 1), i.e. `![](data/figures/<sid>/<name>)`.
- Produces:
  - `class FigureDescriber(Protocol): def describe(self, image_path: Path) -> str: ...`
  - `annotate_figures(md_text: str, *, describer: FigureDescriber, base_dir: Path = Path(".")) -> tuple[str, list[str]]` — returns `(annotated_markdown, figure_paths)`. For each `![](rel)` where `base_dir/rel` `is_probable_figure`, appends `\n\n> **Figure (auto-described):** <desc>\n` after the link and collects `rel`. Non-figures and failed describes are left untouched (path not collected).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_figures.py
from pathlib import Path
from chatqec.ingestion.figures import annotate_figures


class _FakeDescriber:
    def __init__(self, text="a distance-3 surface code lattice"):
        self.text = text
        self.calls = []

    def describe(self, image_path: Path) -> str:
        self.calls.append(Path(image_path))
        return self.text


def test_annotate_injects_marked_description(tmp_path):
    _png(tmp_path, "f.png", (640, 480))
    md = "Intro ![](f.png) end"
    out, figs = annotate_figures(md, describer=_FakeDescriber(), base_dir=tmp_path)
    assert "![](f.png)" in out
    assert "> **Figure (auto-described):** a distance-3 surface code lattice" in out
    assert figs == ["f.png"]


def test_annotate_skips_decorative(tmp_path):
    _png(tmp_path, "logo.png", (16, 16))
    md = "![](logo.png)"
    fake = _FakeDescriber()
    out, figs = annotate_figures(md, describer=fake, base_dir=tmp_path)
    assert out == md and figs == [] and fake.calls == []


def test_annotate_survives_describer_failure(tmp_path):
    _png(tmp_path, "f.png", (640, 480))

    class Boom:
        def describe(self, image_path):
            raise RuntimeError("model down")

    md = "![](f.png)"
    out, figs = annotate_figures(md, describer=Boom(), base_dir=tmp_path)
    assert out == md and figs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_figures.py -q --no-cov`
Expected: FAIL (`cannot import name 'annotate_figures'`)

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/chatqec/ingestion/figures.py
import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

_IMG_REF_RE = re.compile(r"(!\[[^\]]*\]\(([^)]+)\))")


class FigureDescriber(Protocol):
    def describe(self, image_path: Path) -> str: ...


def annotate_figures(
    md_text: str,
    *,
    describer: FigureDescriber,
    base_dir: Path = Path("."),
) -> tuple[str, list[str]]:
    """Append an auto-described blockquote after each real figure link.

    Returns the annotated markdown and the list of figure paths (as they appear
    in the links) that were described. Decorative images, missing files, and
    describer failures leave the markdown unchanged for that reference.
    """
    figures: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        whole, rel = m.group(1), m.group(2)
        path = base_dir / rel
        if not is_probable_figure(path):
            return whole
        try:
            desc = describer.describe(path).strip()
        except Exception:  # never break ingestion on a model error
            logger.warning("figure describe failed for %s", rel, exc_info=True)
            return whole
        if not desc:
            return whole
        figures.append(rel)
        return f"{whole}\n\n> **Figure (auto-described):** {desc}\n"

    return _IMG_REF_RE.sub(_sub, md_text), figures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_figures.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/figures.py tests/test_figures.py
git commit -m "feat(ingest): FigureDescriber protocol + annotate_figures injection"
```

---

### Task 4: Local Llama-3.2-Vision describer

The concrete model. Lazy-loads weights on first use; `# pragma: no cover` (needs GPU + ~20 GB weights). Only the prompt-building helper is unit-tested.

**Files:**
- Create: `src/chatqec/ingestion/figure_vision.py`
- Modify: `pyproject.toml` (optional `vision` extra: `transformers`, `accelerate`, `torch`, `pillow`)
- Test: `tests/test_figure_vision.py`

**Interfaces:**
- Consumes: `FigureDescriber` protocol (Task 3).
- Produces:
  - `FIGURE_PROMPT: str`
  - `class LlamaVisionDescriber:` with `__init__(self, *, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", device: str = "cuda", max_new_tokens: int = 220)` and `describe(self, image_path: Path) -> str`.

- [ ] **Step 1: Write the failing test** (only the prompt contract — model is not loaded)

```python
# tests/test_figure_vision.py
from chatqec.ingestion.figure_vision import FIGURE_PROMPT, LlamaVisionDescriber


def test_prompt_targets_qec_figures():
    p = FIGURE_PROMPT.lower()
    assert "figure" in p
    assert "equation" in p or "label" in p  # asks for OCR-worthy detail


def test_describer_is_a_figuredescriber():
    # Constructing must not load the model (lazy); it only stores config.
    d = LlamaVisionDescriber(device="cpu")
    assert hasattr(d, "describe")
    assert d.model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_figure_vision.py -q --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/chatqec/ingestion/figure_vision.py
"""Local Llama-3.2-Vision figure describer (US-origin, runs on local GPUs)."""

from __future__ import annotations

from pathlib import Path

FIGURE_PROMPT = (
    "You are describing a figure from a quantum error correction paper or "
    "textbook. In 2-4 sentences, state what the figure depicts (code, circuit, "
    "lattice, Tanner graph, decoder, or plot) and transcribe any axis labels, "
    "legends, and equations you can read. Be concrete and technical; no preamble."
)


class LlamaVisionDescriber:
    """Lazy-loading local vision describer implementing ``FigureDescriber``.

    The model (~20 GB) loads on first ``describe`` call, so construction is
    cheap and import-safe on machines without the weights.
    """

    def __init__(
        self,
        *,
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 220,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:  # pragma: no cover - GPU + weights
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def describe(self, image_path: Path) -> str:  # pragma: no cover - GPU + weights
        from PIL import Image

        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": FIGURE_PROMPT}],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(image, prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._processor.decode(out[0], skip_special_tokens=True)
        # Keep only the assistant's continuation after the prompt echo.
        return text.split(FIGURE_PROMPT)[-1].strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_figure_vision.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Add the optional dependency extra**

In `pyproject.toml`, add under `[project.optional-dependencies]`:

```toml
vision = ["transformers>=4.49", "accelerate>=0.34", "torch>=2.2", "pillow>=10"]
```

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/figure_vision.py tests/test_figure_vision.py pyproject.toml
git commit -m "feat(ingest): local Llama-3.2-Vision figure describer (US-origin, GPU)"
```

---

### Task 5: Wire figures into PDF ingestion + carry paths in chunk metadata

**Files:**
- Modify: `src/chatqec/ingestion/fetchers/pdf.py` (`text_to_chunks` — record figure paths per chunk)
- Modify: `src/chatqec/cli.py` (`_ingest_pdf` — compute source_id first, pass `images_dir`, run `annotate_figures`; add `--describe-figures/--no-describe-figures` flag, default on)
- Test: `tests/test_fetcher_pdf.py` (chunk metadata carries figures)

**Interfaces:**
- Consumes: `run_marker_on_pdf(..., images_dir=)` (Task 1), `annotate_figures` (Task 3), `LlamaVisionDescriber` (Task 4), `pdf_id` + suffix logic (existing).
- Produces: chunks whose `metadata["figures"]` is the list of figure paths present in that chunk's text.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fetcher_pdf.py
from chatqec.ingestion.fetchers.pdf import PDFMetadata, text_to_chunks


def test_text_to_chunks_records_figure_paths_in_metadata():
    md = PDFMetadata(content_sha256="a" * 64, title="T", author="A", year=2020, source_path="x.pdf")
    body = "Prose ![](data/figures/pdf_aaaaaaaa/f1.png) more prose."
    chunks = text_to_chunks(md, body_text=body)
    figs = [f for c in chunks for f in c.metadata.get("figures", [])]
    assert "data/figures/pdf_aaaaaaaa/f1.png" in figs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fetcher_pdf.py -k figure -q --no-cov`
Expected: FAIL (metadata has no `figures` key)

- [ ] **Step 3: Implement — record figures per chunk in `text_to_chunks`**

In `pdf.py`, add a module-level regex and populate metadata. Replace the `Chunk(...)` construction so each chunk scans its own text:

```python
import re
_CHUNK_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# inside text_to_chunks, when building each Chunk:
Chunk(
    chunk_id=f"{source_id}#{i}",
    source_id=source_id,
    chunk_index=i,
    text=p,
    title=md.title,
    source_type="pdf",
    authority_tier=tier,
    anchor=f"#chunk-{i}",
    metadata={
        "author": md.author,
        "year": md.year,
        "source_path": md.source_path,
        "figures": _CHUNK_IMG_RE.findall(p),
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fetcher_pdf.py -k figure -q --no-cov`
Expected: PASS

- [ ] **Step 5: Wire `_ingest_pdf` (integration; `# pragma: no cover`)**

In `cli.py` `_ingest_pdf`, before running Marker compute the id, then persist + describe:

```python
from pathlib import Path
from chatqec.ingestion.ids import pdf_id
from chatqec.ingestion.figures import annotate_figures

md = build_pdf_metadata(pdf_path, title=title or pdf_path.stem, author=author or "", year=year or 0)
source_id = pdf_id(md.content_sha256) + suffix
images_dir = Path("data/figures") / source_id.replace(":", "_").replace("/", "_")
body = run_marker_on_pdf(pdf_path, page_range=page_range, images_dir=images_dir)
if describe_figures:
    from chatqec.ingestion.figure_vision import LlamaVisionDescriber
    body, _figs = annotate_figures(body, describer=LlamaVisionDescriber())
chunks = text_to_chunks(md, body_text=body, tier=tier, source_id_suffix=suffix)
```

Add the Click option to the `ingest` command:

```python
@click.option("--describe-figures/--no-describe-figures", default=True,
              help="Run the local vision model to describe extracted figures.")
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add src/chatqec/ingestion/fetchers/pdf.py src/chatqec/cli.py tests/test_fetcher_pdf.py
git commit -m "feat(ingest): describe figures + carry figure paths in chunk metadata"
```

---

## Increment B — Show & reason over figures at answer time (Tasks 6–8)

Produces: retrieved figure chunks load their images into the synthesizer and render to the user. Depends on Increment A's `metadata["figures"]`.

### Task 6: Load figure images for retrieved chunks

**Files:**
- Create: `src/chatqec/retrieval/figure_loader.py`
- Test: `tests/test_figure_loader.py`

**Interfaces:**
- Consumes: `RetrievedChunk.payload`/metadata `figures` list; `ImageAttachment` (`chatqec.api_types`).
- Produces: `load_chunk_figures(chunks: list[RetrievedChunk], *, base_dir: Path = Path("."), max_images: int = 3) -> list[ImageAttachment]` — reads the figure files referenced by the top chunks, returns downscaled `ImageAttachment`s (reuse `ingestion.images.prepare_images`), capped at `max_images`, skipping missing files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figure_loader.py
from pathlib import Path
from PIL import Image
from chatqec.retrieval.hybrid import RetrievedChunk
from chatqec.retrieval.figure_loader import load_chunk_figures


def _chunk(figs):
    return RetrievedChunk(
        chunk_id="c", source_id="pdf:x", text="t", title="T",
        authority_tier="peer_reviewed", score=1.0,
        payload={"figures": figs},
    )


def test_loads_existing_figures_capped(tmp_path):
    for n in ("a.png", "b.png", "c.png", "d.png"):
        Image.new("RGB", (100, 100), (1, 2, 3)).save(tmp_path / n)
    chunk = _chunk(["a.png", "b.png", "c.png", "d.png"])
    imgs = load_chunk_figures([chunk], base_dir=tmp_path, max_images=3)
    assert len(imgs) == 3


def test_skips_missing_files(tmp_path):
    imgs = load_chunk_figures([_chunk(["gone.png"])], base_dir=tmp_path)
    assert imgs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_figure_loader.py -q --no-cov`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/chatqec/retrieval/figure_loader.py
"""Load figure images referenced by retrieved chunks into ImageAttachments."""

from __future__ import annotations

from pathlib import Path

from chatqec.api_types import ImageAttachment
from chatqec.ingestion.images import prepare_images
from chatqec.retrieval.hybrid import RetrievedChunk

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def load_chunk_figures(
    chunks: list[RetrievedChunk],
    *,
    base_dir: Path = Path("."),
    max_images: int = 3,
) -> list[ImageAttachment]:
    """Read up to ``max_images`` figure files referenced by ``chunks``."""
    raw: list[tuple[bytes, str, str]] = []
    seen: set[str] = set()
    for c in chunks:
        for rel in c.payload.get("figures", []) or []:
            if rel in seen:
                continue
            seen.add(rel)
            p = base_dir / rel
            media = _MEDIA.get(p.suffix.lower())
            if media and p.is_file():
                raw.append((p.read_bytes(), media, p.name))
            if len(raw) >= max_images:
                break
        if len(raw) >= max_images:
            break
    if not raw:
        return []
    return prepare_images(raw, max_images=max_images)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_figure_loader.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/retrieval/figure_loader.py tests/test_figure_loader.py
git commit -m "feat(retrieval): load figure images from retrieved chunks"
```

---

### Task 7: Feed retrieved figures to the synthesizer

**Files:**
- Modify: `src/chatqec/api.py` (`ChatQEC.ask` — after retrieval, load figures from top chunks and pass to `synthesize(images=...)`; merge with any user-attached images, cap total at 3)
- Test: `tests/test_api_chatqec.py`

**Interfaces:**
- Consumes: `load_chunk_figures` (Task 6); existing `Synthesizer.synthesize(..., images=)`.
- Produces: `ChatQEC.ask` attaches retrieved figures to synthesis when the user did not attach their own.

- [ ] **Step 1: Write the failing test** (assert synthesizer receives figure images)

```python
# add to tests/test_api_chatqec.py — use the existing fakes/fixtures in that file.
# Build a ChatQEC whose retriever returns a chunk with metadata figures pointing
# at a temp image, and assert the fake synthesizer captured images=[...] of len 1.
# (Follow the construction pattern already used by other tests in this file.)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_chatqec.py -k figure -q --no-cov`
Expected: FAIL (synthesizer received no images)

- [ ] **Step 3: Implement in `api.py` `ask`** — after `canonical_chunks` are built:

```python
from chatqec.retrieval.figure_loader import load_chunk_figures

figure_images = images or load_chunk_figures(retrieval.chunks)
result = self.synthesizer.synthesize(
    question=question,
    ...,
    images=figure_images,
)
```

(Use the real field name for the retrieved chunk list on the retrieval result; wire `images=figure_images` into both `synthesize` and `synthesize_stream` call sites.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api_chatqec.py -k figure -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/api.py tests/test_api_chatqec.py
git commit -m "feat(api): attach retrieved figures to synthesizer context"
```

---

### Task 8: Surface figures to the user (CLI + Streamlit)

**Files:**
- Modify: `src/chatqec/cli.py` (`ask` command — print figure paths used, under a "Figures:" line)
- Modify: `chatqec_app/pages/1_Chat.py` (render figure images beside the answer)
- Test: `tests/test_cli.py` (ask prints a Figures: line when a chunk has figures) — Streamlit render is manual-verify (`# pragma: no cover`).

**Interfaces:**
- Consumes: `Answer` (extend to carry `figure_paths: list[str]` if not present) or read from the retrieval result.

- [ ] **Step 1:** Write a CLI test asserting the `ask` output includes a `Figures:` section listing the figure path when the retrieved chunk has one (use the file's existing CliRunner + fake-provider harness).

- [ ] **Step 2:** Run it — expect FAIL.

- [ ] **Step 3:** Add `figure_paths` to `Answer` (populate from `load_chunk_figures` source paths or the chunks' `metadata["figures"]`), and in the CLI `ask` command print:

```python
if answer.figure_paths:
    click.echo("\nFigures:")
    for p in answer.figure_paths:
        click.echo(f"  {p}")
```

- [ ] **Step 4:** Run it — expect PASS.

- [ ] **Step 5:** In `chatqec_app/pages/1_Chat.py`, after rendering the answer, render figures:

```python
for path in getattr(answer, "figure_paths", []):
    if Path(path).is_file():
        st.image(path, caption=path, use_column_width=True)
```

- [ ] **Step 6:** Full suite + commit.

```bash
.venv/bin/python -m pytest -q
git add src/chatqec/cli.py src/chatqec/api.py chatqec_app/pages/1_Chat.py tests/test_cli.py
git commit -m "feat(ui): show retrieved figures in CLI + Streamlit answers"
```

---

## Task 9: Docs + re-ingest guidance

**Files:**
- Modify: `docs/EXAMPLES.md`, `README.md`

- [ ] **Step 1:** In `README.md`, document `ingest --pdf --describe-figures` and the local vision model + `pip install -e '.[vision]'`.
- [ ] **Step 2:** In `docs/EXAMPLES.md`, add a "Figures" example: ingest a book chapter with figure description, then `chatqec ask "show me the surface-code lattice figure"`.
- [ ] **Step 3:** Note that the 6 already-ingested PDFs must be **re-ingested** (`Raw/` still holds the source PDFs) to gain figures — list the exact re-ingest commands from `sources/registry.jsonl` `notes`.
- [ ] **Step 4:** Commit.

```bash
git add README.md docs/EXAMPLES.md
git commit -m "docs: figure ingestion + re-ingest guidance"
```

---

## Self-Review

**Spec coverage:** persist images (T1) ✓; describe with free local US-origin model (T2–T4) ✓; searchable via embedded text (T3, T5) ✓; paths in metadata (T5) ✓; show to user (T6–T8) ✓; reason over figure at answer time (T7) ✓; re-ingest existing books (T9) ✓.

**Placeholder scan:** Tasks 1–6 carry complete code. Tasks 7–8 reference existing in-file harnesses (fakes in `test_api_chatqec.py`, CliRunner in `test_cli.py`) rather than repeating them — the implementer must read those files; the retrieval-result field name for the chunk list is to be confirmed against `api.py` at implementation time (noted inline).

**Type consistency:** `FigureDescriber.describe(Path)->str` used identically in T3/T4/T6; `annotate_figures -> (str, list[str])`; `load_chunk_figures -> list[ImageAttachment]`; `metadata["figures"]: list[str]` written in T5, read in T6/T8.

**Open confirmation for implementer:** the exact attribute holding retrieved chunks on the retrieval result (`retrieval.chunks` vs other) — verify in `api.py`/`hybrid.py` before Task 7.
```
