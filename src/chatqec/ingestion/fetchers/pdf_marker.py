r"""Isolated-Marker PDF fallback.

For the minority of arXiv papers with no LaTeX source (arXiv serves a PDF),
we extract text with `Marker <https://github.com/datalab-to/marker>`_, which
reconstructs equations to LaTeX and preserves code/tables.

Marker pins ``transformers`` 4.x, which conflicts with this project's
``transformers`` 5.x (used by sentence-transformers). So we never add it to the
project deps — we run it in an **isolated** ``uvx`` environment as a subprocess
and read back the Markdown it writes. The command-building and output-location
logic is unit-tested; the subprocess call itself is a thin wrapper.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# Marker's single-file CLI, run in an ephemeral uvx environment so its
# transformers 4.x never touches the main environment.
_MARKER_PACKAGE = "marker-pdf"
_MARKER_ENTRYPOINT = "marker_single"


class MarkerUnavailableError(RuntimeError):
    """Raised when Marker produced no Markdown (failed or not installed)."""


def build_marker_command(pdf_path: Path, out_dir: Path) -> list[str]:
    """Build the isolated ``uvx`` command that runs Marker on a single PDF."""
    return [
        "uvx",
        "--from",
        _MARKER_PACKAGE,
        _MARKER_ENTRYPOINT,
        str(pdf_path),
        "--output_dir",
        str(out_dir),
        "--output_format",
        "markdown",
    ]


def find_marker_markdown(out_dir: Path) -> Path | None:
    """Locate the Markdown file Marker wrote under ``out_dir`` (nested per doc)."""
    candidates = sorted(out_dir.rglob("*.md"))
    return candidates[0] if candidates else None


def run_marker_on_pdf(
    pdf_path: Path, *, timeout: int = 1200
) -> str:  # pragma: no cover - subprocess + models
    """Run Marker on ``pdf_path`` in isolation and return the extracted Markdown.

    Uses the GPU via ``TORCH_DEVICE=cuda`` when available. Raises
    ``MarkerUnavailableError`` if Marker fails or produces no output.
    """
    env = {**os.environ, "TORCH_DEVICE": os.environ.get("TORCH_DEVICE", "cuda")}
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            subprocess.run(
                build_marker_command(pdf_path, out_dir),
                env=env,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise MarkerUnavailableError(f"marker failed for {pdf_path}: {e}") from e

        md_path = find_marker_markdown(out_dir)
        if md_path is None:
            raise MarkerUnavailableError(f"marker produced no markdown for {pdf_path}")
        return md_path.read_text(encoding="utf-8")
