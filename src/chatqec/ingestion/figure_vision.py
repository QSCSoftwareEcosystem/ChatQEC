"""Local Llama-3.2-Vision figure describer (US-origin, runs on local GPUs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        self._model: Any = None
        self._processor: Any = None

    @staticmethod
    def available() -> bool:
        """True if a local vision backend (torch + CUDA) is usable.

        Cheap probe so callers can skip expensive figure harvesting (PDF download
        + Marker) on machines that could never run the model. Weight availability
        is not checked here — a missing/gated checkpoint surfaces as a per-figure
        describe failure, which callers already tolerate.
        """
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _ensure_loaded(self) -> None:  # pragma: no cover - GPU + weights
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]

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
        # Decode ONLY the newly generated tokens (everything after the prompt),
        # which is echo-proof — slicing on the prompt text left the chat
        # template's "assistant" role header in the output.
        prompt_len = inputs["input_ids"].shape[1]
        generated = out[0][prompt_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()  # type: ignore[no-any-return]
