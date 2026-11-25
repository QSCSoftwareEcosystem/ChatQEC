"""``chatqec bootstrap``: give a fresh install a working corpus index.

A clone has the source registry and the canonical wiki, but no vector index
(``qdrant_data/`` is gitignored). This command starts a local Qdrant if none is
running, then fills the collection in one of two ways:

* ``--snapshot PATH_OR_URL`` restores a full index made with Qdrant's snapshot
  API, the fastest way to get the corpus the paper used.
* otherwise the canonical wiki pages in the repository are chunked and embedded,
  which needs no download and gives definitional answers plus every tool.

An existing, non-empty collection is left alone unless ``--force`` is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import requests
from chatqec_app.qdrant_boot import QdrantBootError, ensure_qdrant

_TIMEOUT = 600  # seconds; restoring a ~60 MB snapshot takes a few seconds


def collection_points(url: str, collection: str) -> int | None:
    """Points in ``collection``, or None when it does not exist."""
    resp = requests.get(f"{url.rstrip('/')}/collections/{collection}", timeout=10)
    if resp.status_code == 404:  # noqa: PLR2004
        return None
    resp.raise_for_status()
    return int(resp.json()["result"].get("points_count") or 0)


def restore_snapshot(url: str, collection: str, source: str) -> int:
    """Restore ``collection`` from a snapshot file or http(s) URL; return its point count.

    A URL is handed to Qdrant to download itself; a local file is uploaded, so it
    works whether or not Qdrant can see this machine's filesystem.
    """
    base = f"{url.rstrip('/')}/collections/{collection}/snapshots"
    if source.startswith(("http://", "https://")):
        resp = requests.put(
            f"{base}/recover?wait=true",
            json={"location": source, "priority": "snapshot"},
            timeout=_TIMEOUT,
        )
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise click.UsageError(f"snapshot file not found: {path}")
        with path.open("rb") as fh:
            resp = requests.post(
                f"{base}/upload?wait=true&priority=snapshot",
                files={"snapshot": (path.name, fh)},
                timeout=_TIMEOUT,
            )
    if not resp.ok:
        raise click.ClickException(f"snapshot restore failed ({resp.status_code}): {resp.text}")
    return collection_points(url, collection) or 0


@click.command("bootstrap")
@click.option(
    "--snapshot",
    default=None,
    help="Qdrant snapshot to restore (file path or http(s) URL) instead of embedding.",
)
@click.option("--force", is_flag=True, help="Rebuild even if the collection already has points.")
@click.pass_context
def bootstrap(ctx: click.Context, snapshot: str | None, force: bool) -> None:
    """Start Qdrant if needed and fill the corpus index for a fresh install."""
    # Lazy: chatqec.cli imports this module while it is still being imported.
    from chatqec.cli import _load_config, canonical_embed  # noqa: PLC0415

    cfg: dict[str, Any] = _load_config()
    url = str(cfg["store"]["qdrant_url"])
    collection = str(cfg["store"]["collection"])
    try:
        state = ensure_qdrant(url=url)
    except QdrantBootError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"qdrant at {url}: {state}")

    existing = collection_points(url, collection)
    if existing and not force:
        click.echo(f"collection {collection!r} already has {existing} points; nothing to do")
        click.echo("(pass --force to rebuild, or --snapshot to replace it)")
        return

    provider = str(cfg.get("embedding", {}).get("provider", "gemini"))
    if snapshot:
        click.echo(f"restoring {collection!r} from {snapshot} ...")
        click.echo(f"restored {restore_snapshot(url, collection, snapshot)} points")
        if provider != "gemini":
            # Both embedders give 768-dim vectors, so a mismatch fails silently:
            # queries embed in one space and search vectors from the other.
            click.echo(
                f"warning: embedding.provider is {provider!r}, but ChatQEC's released "
                "snapshots are embedded with gemini-embedding-001. Set "
                "embedding.provider: gemini unless you built this snapshot yourself "
                f"with {provider!r}.",
                err=True,
            )
        return

    if existing:  # --force without a snapshot: start from an empty collection
        requests.delete(f"{url.rstrip('/')}/collections/{collection}", timeout=60)
    click.echo(f"embedding the canonical wiki with the {provider!r} embedder ...")
    ctx.obj = Path(str(cfg["paths"]["canonical_dir"]))  # what the canonical group passes down
    ctx.invoke(
        canonical_embed,
        chunks_dir=Path("data/chunks"),
        registry_path=Path(str(cfg["paths"]["registry"])),
        stage_only=False,
        qdrant_url=url,
        collection=collection,
        provider=provider,
    )
    click.echo(
        "done. The canonical wiki is indexed; for the full corpus run "
        "`chatqec bootstrap --force --snapshot <file-or-url>`."
    )
