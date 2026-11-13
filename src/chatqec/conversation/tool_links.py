"""Long links produced by tools: keep them out of the model, show them to the user.

QEC Studio links carry a whole session as base64 in the URL fragment (1-10 KB). Sent to
the model they cost thousands of tokens (two example links emptied a local model's
answer) and the model has to copy base64 verbatim, which it can corrupt. So the model
sees a short placeholder, and the CLI and the Chat page print the real links from the
recorded tool output under the answer.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any

LONG_URL_CHARS = 400
PLACEHOLDER = "[link shown to the user under the answer]"


def slim_long_urls(value: Any) -> Any:
    """``value`` with every http(s) string longer than LONG_URL_CHARS replaced."""
    if isinstance(value, str):
        if value.startswith(("https://", "http://")) and len(value) > LONG_URL_CHARS:
            return PLACEHOLDER
        return value
    if isinstance(value, dict):
        return {k: slim_long_urls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [slim_long_urls(v) for v in value]
    return value


def _parse(output: str) -> dict[str, Any]:
    """A recorded tool output (JSON, or the Python repr api.py stores) as a dict."""
    for parse in (json.loads, ast.literal_eval):
        try:
            data = parse(output)
        except (TypeError, ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def links_from_tool_calls(tool_calls: Iterable[Any]) -> list[tuple[str, str]]:
    """(label, url) for every ``url`` a tool returned, including ``examples[*].url``."""
    links: dict[str, str] = {}
    for call in tool_calls:
        data = _parse(getattr(call, "output", "") or "")
        if isinstance(data.get("url"), str) and data["url"].startswith("http"):
            links.setdefault(data["url"], call.name)
        for item in data.get("examples") or []:
            if isinstance(item, dict) and str(item.get("url", "")).startswith("http"):
                links.setdefault(item["url"], str(item.get("title") or call.name))
    return [(label, url) for url, label in links.items()]


__all__ = ["LONG_URL_CHARS", "PLACEHOLDER", "links_from_tool_calls", "slim_long_urls"]
