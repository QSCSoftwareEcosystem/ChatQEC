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
