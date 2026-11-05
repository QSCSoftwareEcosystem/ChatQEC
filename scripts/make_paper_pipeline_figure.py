"""Draw the ChatQEC pipeline figure for the NAACL demo paper (wide, short, vector).

Counts come from the repository at run time (source registry, MCP tool list), so the
figure cannot drift from the system it describes.

    uv run python scripts/make_paper_pipeline_figure.py 6ac52ba22379cb285b6dc6d7/figures/pipeline.pdf
"""

from __future__ import annotations

import collections
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent

# Colour per role; text is always dark on a light fill so the figure prints in greyscale.
FILL = {"in": "#E8EEF7", "ret": "#FFF4D6", "llm": "#E9E3F5", "tool": "#E3F2E7", "out": "#F3F3F3"}
EDGE = "#4A4A4A"


def corpus_counts() -> tuple[int, int, collections.Counter[str]]:
    rows = [json.loads(x) for x in (REPO / "sources/registry.jsonl").read_text().splitlines() if x]
    active = [r for r in rows if r.get("state", "active") == "active"]
    chunks = sum(int(r.get("chunk_count") or 0) for r in active)
    return len(active), chunks, collections.Counter(str(r.get("type")) for r in active)


def tool_count() -> int:
    """Number of tools the installed MCP server exposes (its own interpreter)."""
    py = Path.home() / ".local/share/uv/tools/chatqec-mcp-tools/bin/python"
    out = subprocess.run(
        [str(py), "-I", "-c", "from chatqec_mcp_tools.server import _TOOLS; print(len(_TOOLS))"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def box(ax, x, y, w, h, title, body, role, fs=6.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            fc=FILL[role], ec=EDGE, lw=0.7,
        )
    )  # fmt: skip
    ax.text(
        x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=fs + 0.6, weight="bold"
    )
    ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, both=False):
    style = "<|-|>" if both else "-|>"
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=0.8, color=EDGE)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/images/pipeline.pdf"
    n_src, n_chunks, by_type = corpus_counts()
    n_tools = tool_count()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(6.3, 2.05))  # ACL two-column text width
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Columns: x positions with 0.035 gaps so the arrows between boxes stay visible.
    box(
        ax,
        0.02,
        0.55,
        0.12,
        0.41,
        "Question",
        "on-topic check\nquery rewriting\n(2-3 variants)",
        "in",
    )
    box(
        ax, 0.175, 0.55, 0.215, 0.41, "Evidence",
        f"curated corpus:\n{n_src} sources, {n_chunks:,} chunks\n"
        f"({by_type['canonical_wiki']} reviewed pages)\n"
        "hybrid ranking with\ncanonical-page injection",
        "ret",
    )  # fmt: skip
    box(
        ax, 0.425, 0.55, 0.165, 0.41, "LLM",
        "Claude, Gemini,\nor a local model;\ncites sources or\ncalls tools",
        "llm",
    )  # fmt: skip
    box(
        ax, 0.64, 0.04, 0.345, 0.92, f"MCP tool server ({n_tools} tools)",
        "Simulation: Stim, LightStim, Tsim\n\n"
        "Decoding: PyMatching, Tesseract,\nBP+OSD/LSD, MWPF, MLE, CUDA-Q\n\n"
        "FT primitives and compilation:\nFTPrimitiveBench, Qiskit Clifford+T\nand PBC, Pauli checks\n\n"
        "Hardware and SDKs: IBM Quantum,\nDeltakit, Guppy/Selene",
        "tool",
    )  # fmt: skip
    box(
        ax, 0.175, 0.04, 0.415, 0.40, "Answer",
        "cited answer with source cards, or tool-backed\nnumbers with the exact tool input and output,\n"
        "or a refusal; plus a trace of every step",
        "out",
    )  # fmt: skip

    arrow(ax, 0.145, 0.755, 0.172, 0.755)
    arrow(ax, 0.395, 0.755, 0.422, 0.755)
    arrow(ax, 0.595, 0.755, 0.637, 0.755, both=True)
    arrow(ax, 0.5075, 0.545, 0.5075, 0.445)
    ax.text(0.616, 0.79, "MCP", fontsize=6, ha="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    logger.info("wrote %s (%d sources, %d chunks, %d tools)", out, n_src, n_chunks, n_tools)


if __name__ == "__main__":
    main()
