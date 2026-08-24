"""Start a local Qdrant on demand so the app boots without a separate terminal.

The Chat page dies with ``ResponseHandlingException: [Errno 111] Connection
refused`` when nothing is listening on the store URL. ``ensure_qdrant`` closes
that gap: it probes the URL, and only when the URL is local and a ``qdrant``
binary is on PATH does it spawn one, detached, against the repo's existing
``qdrant_data/`` so the corpus is the same one Docker Compose mounts.

Deliberately conservative. It never stops or reconfigures a Qdrant it did not
start, refuses to autostart anything non-local, and can be switched off with
``CHATQEC_AUTOSTART_QDRANT=0`` (what a hosted deploy should do, since spawning a
database from a web app is wrong there).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:6333"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_PROBE_TIMEOUT = 2.0
_POLL_INTERVAL = 0.5
_HTTP_OK = 200


class QdrantBootError(RuntimeError):
    """Qdrant is unreachable and could not be started automatically."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def store_url(repo_root: Path | None = None) -> str:
    """Resolve the store URL the bot will use: env wins, then conf/config.yaml."""
    env = os.environ.get("CHATQEC_QDRANT_URL")
    if env:
        return env
    cfg = (repo_root or _repo_root()) / "conf" / "config.yaml"
    try:
        loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        return str(loaded["store"]["qdrant_url"])
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return _DEFAULT_URL


def _probe(url: str) -> bool:
    """True when a Qdrant answers at ``url``. Never raises."""
    try:
        resp = requests.get(f"{url.rstrip('/')}/collections", timeout=_PROBE_TIMEOUT)
    except requests.RequestException:
        return False
    return bool(resp.status_code == _HTTP_OK)


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def _spawn(*, binary: str, storage_path: Path, snapshots_path: Path, log_path: Path) -> None:
    """Launch Qdrant detached so it outlives the Streamlit process."""
    storage_path.mkdir(parents=True, exist_ok=True)
    snapshots_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "QDRANT__STORAGE__STORAGE_PATH": str(storage_path),
        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(snapshots_path),
    }
    with log_path.open("ab") as log:
        subprocess.Popen(  # binary comes from shutil.which; no shell, no user input
            [binary],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # the programmatic `setsid`
        )


def ensure_qdrant(
    *,
    url: str | None = None,
    repo_root: Path | None = None,
    timeout: float = 20.0,
    probe: Callable[[str], bool] = _probe,
    spawn: Callable[..., None] = _spawn,
    binary: str | None = "",
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Make sure a Qdrant is reachable, starting a local one if needed.

    Returns "disabled", "already-running", or "started".

    Args:
        url: Store URL to probe. Defaults to the resolved config value.
        repo_root: Repo root used for qdrant_data/ and logs/. Defaults to this file's repo.
        timeout: Seconds to wait for a spawned Qdrant to answer.
        probe: Health check, injected in tests.
        spawn: Process launcher, injected in tests.
        binary: Path to the qdrant binary; "" means look it up on PATH, None means absent.
        sleep: Delay function, injected in tests.

    Raises:
        QdrantBootError: Not reachable and autostart was impossible or timed out.
    """
    if os.environ.get("CHATQEC_AUTOSTART_QDRANT", "1") == "0":
        return "disabled"

    url = url or store_url(repo_root)
    if probe(url):
        return "already-running"

    if not _is_local(url):
        raise QdrantBootError(
            f"No Qdrant answering at {url} and it is not local, so it cannot be started here. "
            "Start it on that host, or point store.qdrant_url at a running instance."
        )

    root = repo_root or _repo_root()
    storage = root / "qdrant_data" / "storage"
    snapshots = root / "qdrant_data" / "snapshots"
    log_path = root / "logs" / "qdrant.log"

    resolved = shutil.which("qdrant") if binary == "" else binary
    if not resolved:
        raise QdrantBootError(
            f"No Qdrant at {url} and no `qdrant` binary on PATH. Start it yourself:\n"
            "  docker compose up -d\n"
            "or install the native binary and run:\n"
            f'  QDRANT__STORAGE__STORAGE_PATH="{storage}" \\\n'
            f'    QDRANT__STORAGE__SNAPSHOTS_PATH="{snapshots}" \\\n'
            "    setsid nohup qdrant > logs/qdrant.log 2>&1 &"
        )

    logger.info("No Qdrant at %s; starting %s (log: %s)", url, resolved, log_path)
    spawn(binary=resolved, storage_path=storage, snapshots_path=snapshots, log_path=log_path)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe(url):
            logger.info("Qdrant is up at %s", url)
            return "started"
        sleep(_POLL_INTERVAL)

    raise QdrantBootError(
        f"Started {resolved} but nothing answered at {url} within {timeout:g}s. "
        f"Check {log_path} — a stale lock on {storage} or a port clash on 6333 is the usual cause."
    )
