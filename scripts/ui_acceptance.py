"""Drive the running Streamlit Chat page with Playwright and record each answer.

Usage:
    uv run --with playwright python scripts/ui_acceptance.py \
        --port 8501 --label muse --prompts eval/ui_acceptance_prompts.json --out logs/ui

Each prompt is typed into the real chat input, so this exercises the same path a
user does (off-topic gate, router, retrieval, MCP tools, rendering). For every
prompt it writes a full-page screenshot and one JSON row: status, seconds, tool
names shown in the trace, table and exception counts, and the head of the answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

logger = logging.getLogger(__name__)

RUN_START_TIMEOUT_S = 30
ANSWER_TIMEOUT_S = 600
IDLE_SETTLE_S = 4
MIN_ANSWER_CHARS = 40  # shorter text is a placeholder, not an answer


def _running(page: Page) -> bool:
    return (
        page.locator('[data-testid="stStatusWidget"]').count() > 0
        or page.get_by_role("button", name="Stop").count() > 0
        or "Running QEC tools" in page.inner_text("body")
    )


def _ask(page: Page, port: int, prompt: str) -> tuple[str, float]:
    """Submit one prompt and wait until a run has started and then settled."""
    page.goto(f"http://localhost:{port}/Chat")
    page.wait_for_selector('[data-testid="stChatInput"] textarea', timeout=90_000)
    page.wait_for_timeout(2500)
    box = page.locator('[data-testid="stChatInput"] textarea')
    box.fill(prompt)
    box.press("Enter")
    t0 = time.time()

    started = False
    while time.time() - t0 < RUN_START_TIMEOUT_S:
        if _running(page):
            started = True
            break
        page.wait_for_timeout(500)

    idle_since: float | None = None
    while time.time() - t0 < ANSWER_TIMEOUT_S:
        if _running(page):
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            last = page.locator('[data-testid="stChatMessage"]').last
            settled = time.time() - idle_since >= IDLE_SETTLE_S
            if started and settled and len(last.inner_text()) > MIN_ANSWER_CHARS:
                return "ok", time.time() - t0
        page.wait_for_timeout(700)
    return ("timeout" if started else "run never started"), time.time() - t0


def run(
    port: int,
    label: str,
    prompts: list[dict[str, Any]],
    out: Path,
    executable_path: str | None = None,
) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, executable_path=executable_path, args=["--no-sandbox"]
        )
        for item in prompts:
            page = browser.new_page(viewport={"width": 1300, "height": 1100})
            status, seconds = _ask(page, port, item["prompt"])
            msgs = page.locator('[data-testid="stChatMessage"]')
            answer = msgs.last.inner_text() if msgs.count() else ""
            tools = [
                ln.split("Tool:")[1].strip().strip("`")
                for ln in answer.splitlines()
                if ln.strip().startswith("Tool:")
            ]
            row = {
                "label": label,
                "name": item["name"],
                "status": status,
                "seconds": round(seconds),
                "expected_tools": item.get("expected_tools", []),
                "tools_shown": tools,
                "tables": page.locator('[data-testid="stDataFrame"]').count(),
                "exceptions": page.locator('[data-testid="stException"]').count(),
                "answer_head": answer[:700],
            }
            page.screenshot(path=str(out / f"ui_{label}_{item['name']}.png"), full_page=True)
            rows.append(row)
            logger.info(json.dumps({k: v for k, v in row.items() if k != "answer_head"}))
            page.close()
        browser.close()
    (out / f"ui_{label}.json").write_text(json.dumps(rows, indent=1))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--label", required=True, help="tag for output files, e.g. the model name")
    ap.add_argument("--prompts", type=Path, default=Path("eval/ui_acceptance_prompts.json"))
    ap.add_argument("--out", type=Path, default=Path("logs/ui"))
    ap.add_argument(
        "--executable-path",
        default=None,
        help="Chromium binary to use when the cached browser build differs from Playwright's",
    )
    args = ap.parse_args()
    prompts = json.loads(args.prompts.read_text())
    run(args.port, args.label, prompts, args.out, args.executable_path)


if __name__ == "__main__":
    main()
