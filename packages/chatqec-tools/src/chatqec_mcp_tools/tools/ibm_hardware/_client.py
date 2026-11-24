"""Lazy IBM Quantum Platform authentication.

Verified live 2026-09-03 against a free/open-plan account: only the
"ibm_quantum_platform" channel exists in current qiskit-ibm-runtime (the legacy
"ibm_quantum" channel raises ValueError), and no instance is required -- the SDK
auto-selects the account's single open-plan instance when none is passed.

That auto-select only works once the account has at least one provisioned
Cloud service instance, though. A token that authenticates fine but is tied to
an account with zero instances makes QiskitRuntimeService.__init__ raise
IBMInputValueError("No matching instances found..."). Verified live
2026-09-15: a syntactically valid, non-expired token with zero linked
instances reproduces this exactly (confirmed directly via
Account.list_instances() returning []). get_ibm_service below matches on that
message rather than importing IBMInputValueError itself, so this stays a
plain string check instead of a second hard import -- this module's whole
qiskit_ibm_runtime import already stays lazy and inside try/except so a base
install without the 'ibm' extra never touches it.
"""

from __future__ import annotations

import os
from typing import Any

IBM_CHANNEL = "ibm_quantum_platform"

# The one canonical install hint for the optional 'ibm' extra. Shared so every
# tool's "SDK missing" message reads identically -- map_circuit.compute_mapping
# reuses it for the qiskit (not qiskit-ibm-runtime) import path.
IBM_EXTRA_INSTALL_HINT = (
    "Install the 'ibm' extra: uv tool install --force '~/chatqec-mcp-tools[ibm]'"
)


class IBMCredentialsError(RuntimeError):
    """Raised when IBM Quantum credentials are missing, invalid, or the SDK extra
    is not installed. Every ibm_hardware tool catches this and returns
    ok=False + error_message rather than letting it propagate."""


def get_ibm_service() -> Any:
    """Build a qiskit_ibm_runtime.QiskitRuntimeService from QISKIT_IBM_TOKEN /
    QISKIT_IBM_INSTANCE env vars. Never raises anything but IBMCredentialsError."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: PLC0415
    except ImportError as exc:
        raise IBMCredentialsError(
            f"qiskit-ibm-runtime is not installed. {IBM_EXTRA_INSTALL_HINT}"
        ) from exc

    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")

    kwargs: dict[str, Any] = {"channel": IBM_CHANNEL}
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance

    try:
        return QiskitRuntimeService(**kwargs)
    except Exception as exc:
        if "No matching instances found" in str(exc) and not instance:
            raise IBMCredentialsError(
                "This IBM Cloud account has no provisioned Quantum service "
                "instance (the token itself is valid). Create one (e.g. the "
                "free Open Plan) at https://quantum.cloud.ibm.com/, or if one "
                "already exists, set QISKIT_IBM_INSTANCE to its CRN or name."
            ) from exc
        raise IBMCredentialsError(f"Could not authenticate with IBM Quantum: {exc}") from exc
