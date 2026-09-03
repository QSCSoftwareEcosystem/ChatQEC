from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_qiskit_ibm_runtime(monkeypatch, *, service_factory=None, raise_on_import=False):
    if raise_on_import:
        monkeypatch.delitem(sys.modules, "qiskit_ibm_runtime", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _raising_import("qiskit_ibm_runtime"),
        )
        return

    fake_module = ModuleType("qiskit_ibm_runtime")
    fake_module.QiskitRuntimeService = service_factory or (
        lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    )
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_module)


def _raising_import(blocked_name):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == blocked_name or name.startswith(blocked_name + "."):
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)

    return fake_import


def test_get_ibm_service_uses_token_from_env(monkeypatch):
    from chatqec_mcp_tools.tools.ibm_hardware._client import get_ibm_service

    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    _install_fake_qiskit_ibm_runtime(monkeypatch, service_factory=factory)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "fake-token-value")
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)

    get_ibm_service()

    assert captured == {"channel": "ibm_quantum_platform", "token": "fake-token-value"}


def test_get_ibm_service_includes_instance_when_set(monkeypatch):
    from chatqec_mcp_tools.tools.ibm_hardware._client import get_ibm_service

    captured = {}
    _install_fake_qiskit_ibm_runtime(
        monkeypatch, service_factory=lambda **kw: captured.update(kw) or SimpleNamespace()
    )
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "fake-token-value")
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", "crn:v1:fake-instance")

    get_ibm_service()

    assert captured["instance"] == "crn:v1:fake-instance"


def test_get_ibm_service_raises_credentials_error_without_extra(monkeypatch):
    from chatqec_mcp_tools.tools.ibm_hardware._client import (
        IBMCredentialsError,
        get_ibm_service,
    )

    _install_fake_qiskit_ibm_runtime(monkeypatch, raise_on_import=True)

    with pytest.raises(IBMCredentialsError, match="qiskit-ibm-runtime"):
        get_ibm_service()


def test_get_ibm_service_wraps_sdk_auth_failure(monkeypatch):
    from chatqec_mcp_tools.tools.ibm_hardware._client import (
        IBMCredentialsError,
        get_ibm_service,
    )

    def raising_factory(**kwargs):
        raise RuntimeError("Unable to retrieve instances. Please check your token.")

    _install_fake_qiskit_ibm_runtime(monkeypatch, service_factory=raising_factory)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "bad-token")

    with pytest.raises(IBMCredentialsError, match="Unable to retrieve instances"):
        get_ibm_service()
