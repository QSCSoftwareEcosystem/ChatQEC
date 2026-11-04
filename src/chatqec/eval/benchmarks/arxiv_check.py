"""Does an arXiv id exist? Cached on disk so citation verdicts are reproducible.

Used to tell a real paper a model cites from one it invented. The cache file is committed with
the comparison results; a run never changes a verdict already in it.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

_API = "https://export.arxiv.org/api/query?id_list={}"
_ENTRY_ID = re.compile(r"<entry>.*?<id>https?://arxiv\.org/abs/([^<v]+)", re.DOTALL)


def _fetch(arxiv_id: str) -> str:
    req = urllib.request.Request(_API.format(arxiv_id), headers={"User-Agent": "chatqec-eval"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return str(resp.read().decode("utf-8", errors="replace"))


def exists_from_feed(feed: str, arxiv_id: str) -> bool:
    """arXiv answers an unknown id with an entry titled 'Error'; a real one echoes its id."""
    return any(found.strip() == arxiv_id for found in _ENTRY_ID.findall(feed))


def cached_checker(cache: Path, fetch: Callable[[str], str] = _fetch) -> Callable[[str], bool]:
    known: dict[str, bool] = json.loads(cache.read_text()) if cache.exists() else {}

    def exists(arxiv_id: str) -> bool:
        if arxiv_id not in known:
            known[arxiv_id] = exists_from_feed(fetch(arxiv_id), arxiv_id)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
        return known[arxiv_id]

    return exists


def corpus_arxiv_ids(registry: Path) -> set[str]:
    ids = set()
    for line in registry.read_text().splitlines():
        sid = str(json.loads(line).get("id", ""))
        if sid.startswith("arxiv:"):
            ids.add(sid.removeprefix("arxiv:"))
    return ids
