"""Capture paper-quality screenshots of the ChatQEC Chat page for the demo paper.

Drives a running Streamlit app with Playwright (same submit-and-wait logic as
``scripts/ui_acceptance.py``), at a 1600 px wide viewport and 2x device scale so the figures
stay sharp in print. For each scenario it expands the first citation card and saves:

- ``<name>_answer.png``: the assistant message (answer, citation cards, tool tables, trace)
- ``<name>_page.png``: the full page, for the appendix

Usage (app started with CHATQEC_PROVIDER=local CHATQEC_LOCAL_MODEL=muse-glimmer
CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run streamlit run chatqec_app/streamlit_app.py --server.port 8503):

    uv run --with playwright python scripts/capture_screenshots.py --port 8503 \\
        --out 6ac52ba22379cb285b6dc6d7/figures/screens \\
        --executable-path ~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from ui_acceptance import _ask

logger = logging.getLogger(__name__)

# Scenario prompts come from committed eval files so the paper shows tested questions.
SCENARIOS = {
    "definition": ("eval/ui_acceptance_prompts.json", "surface_code_definition"),
    "computation": ("eval/ui_acceptance_prompts.json", "decoder_compare_color"),
    "refusal": ("eval/compare/prompts.jsonl", "abstain-willow-d15"),
}


def _prompt(source: str, key: str) -> str:
    path = Path(source)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        row = next(r for r in rows if r.get("id") == key)
        return str(row.get("prompt") or row["question"])
    row = next(r for r in json.loads(path.read_text()) if r["name"] == key)
    return str(row["prompt"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8503)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="scenario names to capture")
    ap.add_argument("--executable-path", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, executable_path=args.executable_path, args=["--no-sandbox"]
        )
        for name, (source, key) in SCENARIOS.items():
            if args.only and name not in args.only:
                continue
            page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=2)
            status, seconds = _ask(page, args.port, _prompt(source, key))
            answer = page.locator('[data-testid="stChatMessage"]').last
            cards = answer.locator('[data-testid="stExpander"] summary')
            if cards.count():
                cards.first.click()
                page.wait_for_timeout(800)
            # Hide the sticky chat input (it overlays the bottom of tall messages), and make the
            # viewport taller than the message: Streamlit scrolls an inner container, so parts
            # of a message outside the viewport otherwise render blank or get clipped.
            page.add_style_tag(content='[data-testid="stBottom"]{display:none !important}')
            box = answer.bounding_box()
            if box is None:
                raise RuntimeError(f"{name}: assistant message not found")
            page.set_viewport_size({"width": 1600, "height": int(box["height"]) + 600})
            page.wait_for_timeout(800)
            answer.scroll_into_view_if_needed()
            answer.screenshot(path=str(args.out / f"{name}_answer.png"))
            page.screenshot(path=str(args.out / f"{name}_page.png"), full_page=True)
            logger.info("%s: %s in %.0f s, %d expanders", name, status, seconds, cards.count())
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
