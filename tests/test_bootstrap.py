"""chatqec bootstrap helpers: collection probing and snapshot restore (HTTP faked)."""

from types import SimpleNamespace

import click
import pytest

from chatqec import bootstrap as bs

URL = "http://localhost:6333"


def _resp(status=200, payload=None, text=""):
    return SimpleNamespace(
        status_code=status,
        ok=status < 400,
        text=text,
        json=lambda: payload,
        raise_for_status=lambda: None,
    )


def test_collection_points_missing_collection_is_none(monkeypatch):
    monkeypatch.setattr(bs.requests, "get", lambda *a, **k: _resp(404))
    assert bs.collection_points(URL, "c") is None


def test_collection_points_reads_count(monkeypatch):
    monkeypatch.setattr(
        bs.requests, "get", lambda *a, **k: _resp(payload={"result": {"points_count": 314}})
    )
    assert bs.collection_points(URL, "c") == 314


def test_restore_from_url_asks_qdrant_to_download(monkeypatch):
    calls = {}

    def fake_put(url, json, timeout):
        calls["url"], calls["json"] = url, json
        return _resp()

    monkeypatch.setattr(bs.requests, "put", fake_put)
    monkeypatch.setattr(bs, "collection_points", lambda *a: 3266)
    n = bs.restore_snapshot(URL, "c", "https://example.org/x.snapshot")
    assert n == 3266
    assert calls["url"] == f"{URL}/collections/c/snapshots/recover?wait=true"
    assert calls["json"]["location"] == "https://example.org/x.snapshot"


def test_restore_from_file_uploads_it(monkeypatch, tmp_path):
    snap = tmp_path / "x.snapshot"
    snap.write_bytes(b"data")
    seen = {}

    def fake_post(url, files, timeout):
        seen["url"] = url
        seen["name"], fh = files["snapshot"]
        seen["body"] = fh.read()
        return _resp()

    monkeypatch.setattr(bs.requests, "post", fake_post)
    monkeypatch.setattr(bs, "collection_points", lambda *a: 7)
    assert bs.restore_snapshot(URL, "c", str(snap)) == 7
    assert seen == {
        "url": f"{URL}/collections/c/snapshots/upload?wait=true&priority=snapshot",
        "name": "x.snapshot",
        "body": b"data",
    }


def test_restore_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(click.UsageError):
        bs.restore_snapshot(URL, "c", str(tmp_path / "nope.snapshot"))


def test_restore_failure_reports_qdrant_message(monkeypatch):
    monkeypatch.setattr(bs.requests, "put", lambda *a, **k: _resp(400, text="bad snapshot"))
    with pytest.raises(click.ClickException, match="bad snapshot"):
        bs.restore_snapshot(URL, "c", "https://example.org/x.snapshot")
