from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _fake_backend(name, num_qubits, operational, pending_jobs, basis_gates):
    return SimpleNamespace(
        name=name,
        num_qubits=num_qubits,
        status=lambda: SimpleNamespace(operational=operational, pending_jobs=pending_jobs),
        operation_names=basis_gates,
    )


def test_ibm_list_backends_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
        IbmListBackendsInput,
        ibm_list_backends,
    )

    fake_service = SimpleNamespace(
        backends=lambda: [
            _fake_backend("ibm_fez", 156, True, 0, ["cz", "sx", "x", "rz"]),
            _fake_backend("ibm_kingston", 156, True, 3, ["cz", "sx", "x", "rz"]),
        ]
    )

    with patch(
        "chatqec_mcp_tools.tools.ibm_hardware.list_backends.get_ibm_service",
        return_value=fake_service,
    ):
        out = ibm_list_backends(IbmListBackendsInput())

    assert out.ok is True
    assert [b.name for b in out.backends] == ["ibm_fez", "ibm_kingston"]
    assert out.backends[0].num_qubits == 156
    assert out.backends[1].pending_jobs == 3


def test_ibm_list_backends_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError
    from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
        IbmListBackendsInput,
        ibm_list_backends,
    )

    with patch(
        "chatqec_mcp_tools.tools.ibm_hardware.list_backends.get_ibm_service",
        side_effect=IBMCredentialsError("no token configured"),
    ):
        out = ibm_list_backends(IbmListBackendsInput())

    assert out.ok is False
    assert "no token configured" in out.error_message
    assert out.backends == []


# --------------------------------------------------------------------------
# A cached payload whose SHAPE does not match must degrade, not raise.
# --------------------------------------------------------------------------


def _write_cache(tmp_path, key, payload):
    """Write a cache entry directly, mimicking _cache.get_or_fetch's on-disk
    format with a TTL far in the future so it is served without refetching."""
    import json
    import time

    root = tmp_path / "chatqec-mcp-tools" / "ibm_hardware"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{key.replace('/', '__')}.json").write_text(
        json.dumps({"fetched_at": time.time(), "ttl_seconds": 1e9, "payload": payload}),
        encoding="utf-8",
    )


def test_ibm_list_backends_handles_cached_payload_shape_mismatch(monkeypatch, tmp_path):
    """The model_validate calls used to sit OUTSIDE the try/except wrapping
    get_or_fetch, so a cache file that is valid JSON but the wrong shape (a
    future schema change, a hand-edited file, or the same cache key written by
    map_circuit.py's auto-select via a slightly different code path) raised
    pydantic's ValidationError straight out of the tool -- violating the
    feature-wide "every failure path returns ok=False, nothing raises"
    contract."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
        IbmListBackendsInput,
        ibm_list_backends,
    )

    # Right key, fresh entry, valid JSON -- wrong shape for BackendInfo.
    _write_cache(tmp_path, "list_backends", [{"name": "ibm_fake", "qubit_count": 5}])

    out = ibm_list_backends(IbmListBackendsInput())

    assert out.ok is False
    assert out.backends == []
    assert "unexpected format" in out.error_message
    assert "retry" in out.error_message.lower()


def test_ibm_list_backends_handles_non_list_cached_payload(monkeypatch, tmp_path):
    """Same guard, different corruption: a payload that is not even iterable as
    a list of dicts must produce ok=False rather than a TypeError."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
        IbmListBackendsInput,
        ibm_list_backends,
    )

    _write_cache(tmp_path, "list_backends", [None, 42])

    out = ibm_list_backends(IbmListBackendsInput())

    assert out.ok is False
    assert "unexpected format" in out.error_message
