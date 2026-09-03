# IBM Hardware-Aware Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five MCP tools to `chatqec-mcp-tools` that let ChatQEC list IBM Quantum
backends, answer calibration/noise questions with cache-aware freshness, map an arbitrary
user-supplied circuit (QASM or a named preset) onto a chosen or auto-selected backend using
an adapted version of the intern's quality-scored placement algorithms, and optionally submit
a capped, two-step-confirmed real hardware job.

**Architecture:** All five tools live in a new `tools/ibm_hardware/` subpackage inside
`chatqec-mcp-tools`, following that repo's existing one-file-per-tool / pydantic
`Input`/`Output` / `ok`+`error_message` conventions. `qiskit`/`qiskit-ibm-runtime` are lazy
imports behind a new `ibm` extra. A new file-based TTL cache avoids re-hitting the IBM API on
every chat turn. The mapping engine adapts (not copies) three modules from
`/home/pza/intern/sangram/HardwareAwarenessIntegrationPipeline/cahgr/`: the hardware graph
schema, the objective-based quality model, and greedy+VF2 placement — built from a live SDK
backend object instead of a parsed JSON snapshot, and with VF2's hardcoded node/candidate
caps replaced by a wall-clock time budget (the one confirmed real defect in the source).

**Tech Stack:** Python ≥3.11, `qiskit>=1.0`, `qiskit-ibm-runtime>=0.48` (new optional `ibm`
extra), `networkx` (already a dependency), `matplotlib` (already a dependency, for the
coupling-map SVG), pydantic v2, pytest.

**Spec:** `docs/planning/specs/2026-09-03-ibm-hardware-mapping-design.md`

## Global Constraints

- **Repo:** everything in this plan modifies `/home/pza/chatqec-mcp-tools`, not `chatQEC`.
  No branch (repo convention observed: work lands on `main`, per this repo's existing
  history) unless told otherwise.
- **Every tool function** has the shape `def <tool_name>(inp: <Name>Input) -> <Name>Output`,
  no decorators, registered in `src/chatqec_mcp_tools/server.py`'s `_TOOLS` list via a thin
  `_<tool>_handler(args: dict) -> dict` doing `Input.model_validate(args)` → call →
  `.model_dump()`. Every `Input`/`Output` model has `model_config = ConfigDict(extra="forbid")`.
  Every failure path returns `ok=False` + `error_message`; **nothing raises out of a tool
  function.**
- **Lazy imports.** `qiskit`, `qiskit_ibm_runtime` are imported inside function bodies only,
  never at module top, so the package still imports without the `ibm` extra installed
  (matches the existing `stim`/`pymatching` pattern in this repo).
- **IBM channel:** `"ibm_quantum_platform"` only. Verified live 2026-09-03: the legacy
  `ibm_quantum` channel no longer exists in `qiskit-ibm-runtime` 0.49+ and raises `ValueError`.
  No `QISKIT_IBM_INSTANCE` is required on a free/open-plan account — omit `instance` from the
  `QiskitRuntimeService` kwargs entirely when the env var is unset, don't pass `instance=None`
  as an empty override (the SDK auto-selects `open-instance` and only warns).
- **Cache location:** `$XDG_CACHE_HOME/chatqec-mcp-tools/ibm_hardware/` (fallback
  `~/.cache/...`), one JSON file per cache key. Default TTLs: `60` seconds for backend
  list/status, `3600` seconds for calibration (both overridable via
  `CHATQEC_IBM_BACKEND_LIST_TTL_SECONDS` / `CHATQEC_IBM_CALIBRATION_TTL_SECONDS`).
- **Credentials:** `QISKIT_IBM_TOKEN` / `QISKIT_IBM_INSTANCE` env vars only. Never a tool
  input parameter, never written to any file.
- **Execution safety caps** (enforced in `ibm_submit_execution` only, checked before any SDK
  call): `num_qubits <= 10`, `shots <= 4000`, `backend` is required (no auto-pick allowed for
  submission — auto-pick is fine for `ibm_map_circuit`/`ibm_preview_execution`).
- **VF2 fix:** replace `cahgr/retrieval.py`'s `_VF2_NODE_CAP = 12` / `_VF2_CANDIDATE_CAP =
  4000` hardcoded constants with a wall-clock time budget (`_VF2_TIME_BUDGET_SECONDS = 2.0`,
  checked every loop iteration via `time.monotonic()`), keeping a generous node-count
  pre-filter (`_VF2_MAX_INTERACTION_NODES = 20`, purely to skip an obviously-too-large
  interaction graph before starting the search) but no candidate-count cap — the time budget
  is the real termination guarantee.
- **Do not port** `hardware_layout.vendor_ibm` (JSON reconstruction) or
  `hardware_layout.qubit_coordinates` (per-chip coordinate tables) — `_hardware_graph.py`
  builds directly from a live backend's `.properties()` / `.coupling_map`, and the SVG uses
  `networkx.kamada_kawai_layout` for node positions instead.
- **Do not port** `qec/five_qubit_code.py`, `llm/circuit_generation.py`, `cahgr/benchmark.py`,
  `cahgr/layout_search.py`/`llm_optimizer.py`, or Steiner-tree placement — out of scope per
  the spec.
- Python: 4-space indent, type hints on public functions, `from __future__ import
  annotations`, matching the existing repo style.
- Commit after every task (once its own tests are green) — do not batch multiple tasks into
  one commit. No `Co-Authored-By` trailer. Conventional Commits format.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | New `ibm` optional-dependency extra: `qiskit>=1.0`, `qiskit-ibm-runtime>=0.48`. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/__init__.py` (new) | Empty, matches existing `tools/__init__.py` convention. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_client.py` (new) | Lazy `QiskitRuntimeService` construction from env vars. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_cache.py` (new) | File-based TTL cache, new pattern for this repo. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/list_backends.py` (new) | `ibm_list_backends` tool. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_hardware_graph.py` (new) | Live backend → normalized `networkx.Graph`, adapted from `cahgr/hardware_kg.py`. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/calibration.py` (new) | `ibm_backend_calibration` tool. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_quality_model.py` (new) | `HardwareQualityModel`, ported from `cahgr/retrieval.py`. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py` (new) | Greedy + time-budgeted VF2 placement, adapted from `cahgr/retrieval.py`. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_circuits.py` (new) | QASM parsing + bell/ghz/qft/qaoa presets + interaction-graph builder, adapted from `cahgr/circuit_graph.py` + `qec/prompt_library.py`. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/_coupling_diagram.py` (new) | Coupling-map → SVG string, mapped qubits highlighted. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py` (new) | `ibm_map_circuit` tool. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/preview_execution.py` (new) | `ibm_preview_execution` tool. |
| `src/chatqec_mcp_tools/tools/ibm_hardware/submit_execution.py` (new) | `ibm_submit_execution` tool. |
| `src/chatqec_mcp_tools/server.py` (modify) | Register all 5 new `ToolSpec` entries. |
| `tests/test_ibm_*.py` (new, one per tool/module) | Unit tests, no live network. |
| `tests/test_server.py` (modify) | Registration assertions for the 5 new tools. |
| `README.md` (modify) | Document the new tools, the `ibm` extra, and the two required env vars. |

---

### Task 1: Package scaffold + `ibm` extra + `_client.py`

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/__init__.py`
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_client.py`
- Modify: `pyproject.toml`
- Test: `tests/test_ibm_client.py`

**Interfaces:**
- Produces: `IBMCredentialsError(RuntimeError)`, `get_ibm_service() -> Any` (a
  `qiskit_ibm_runtime.QiskitRuntimeService` instance), `IBM_CHANNEL: str` constant. Every
  later task that talks to IBM calls `get_ibm_service()` and catches `IBMCredentialsError`.

- [ ] **Step 1: Add the `ibm` extra to `pyproject.toml`**

Find the `[project.optional-dependencies]` table (alongside the existing `dev`, `lightstim`,
`tsim` groups) and add:

```toml
ibm = ["qiskit>=1.0", "qiskit-ibm-runtime>=0.48"]
```

- [ ] **Step 2: Create the empty package `__init__.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/__init__.py
```

(Empty file, matching the existing `tools/__init__.py` convention of not re-exporting.)

- [ ] **Step 3: Write the failing test for `_client.py`**

```python
# tests/test_ibm_client.py
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/test_ibm_client.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'chatqec_mcp_tools.tools.ibm_hardware'`

- [ ] **Step 5: Implement `_client.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_client.py
"""Lazy IBM Quantum Platform authentication.

Verified live 2026-09-03 against a free/open-plan account: only the
"ibm_quantum_platform" channel exists in current qiskit-ibm-runtime (the legacy
"ibm_quantum" channel raises ValueError), and no instance is required -- the SDK
auto-selects the account's single open-plan instance when none is passed.
"""

from __future__ import annotations

import os
from typing import Any

IBM_CHANNEL = "ibm_quantum_platform"


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
            "qiskit-ibm-runtime is not installed. Install the 'ibm' extra: "
            "uv tool install --force '~/chatqec-mcp-tools[ibm]'"
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
        raise IBMCredentialsError(
            f"Could not authenticate with IBM Quantum: {exc}"
        ) from exc
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_ibm_client.py -v`
Expected: `PASS` (4 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/chatqec_mcp_tools/tools/ibm_hardware/__init__.py \
        src/chatqec_mcp_tools/tools/ibm_hardware/_client.py tests/test_ibm_client.py
git commit -m "feat(ibm): add ibm extra and lazy IBM Quantum auth client"
```

---

### Task 2: `_cache.py` — file-based TTL cache

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_cache.py`
- Test: `tests/test_ibm_cache.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `CacheResult` (dataclass: `payload: Any`, `fetched_at: float`, `age_seconds:
  float`, `stale: bool`), `get_or_fetch(key: str, ttl_seconds: float, fetch_fn: Callable[[],
  Any]) -> CacheResult`. Consumed by Tasks 4, 6, 12, 13.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_cache.py
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield tmp_path


def test_cache_miss_calls_fetch_fn_and_persists():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    calls = []

    def fetch():
        calls.append(1)
        return {"value": 42}

    result = get_or_fetch("k1", ttl_seconds=60, fetch_fn=fetch)

    assert result.payload == {"value": 42}
    assert result.age_seconds == 0.0
    assert result.stale is False
    assert len(calls) == 1

    result2 = get_or_fetch("k1", ttl_seconds=60, fetch_fn=fetch)
    assert result2.payload == {"value": 42}
    assert len(calls) == 1  # second call served from cache, fetch_fn not called again


def test_cache_expiry_triggers_refetch():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    get_or_fetch("k2", ttl_seconds=0, fetch_fn=fetch)
    time.sleep(0.01)
    result = get_or_fetch("k2", ttl_seconds=0, fetch_fn=fetch)

    assert len(calls) == 2
    assert result.payload == {"n": 2}


def test_corrupt_cache_file_is_treated_as_miss(tmp_path):
    from chatqec_mcp_tools.tools.ibm_hardware._cache import _cache_path, get_or_fetch

    path = _cache_path("k3")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{{", encoding="utf-8")

    result = get_or_fetch("k3", ttl_seconds=60, fetch_fn=lambda: {"ok": True})

    assert result.payload == {"ok": True}
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == {"ok": True}


def test_stale_cache_served_when_fetch_fails():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    get_or_fetch("k4", ttl_seconds=0, fetch_fn=lambda: {"v": "first"})

    def failing_fetch():
        raise RuntimeError("network down")

    result = get_or_fetch("k4", ttl_seconds=0, fetch_fn=failing_fetch)

    assert result.payload == {"v": "first"}
    assert result.stale is True


def test_fetch_failure_with_no_cache_raises():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    def failing_fetch():
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        get_or_fetch("k5", ttl_seconds=60, fetch_fn=failing_fetch)


def test_keys_are_isolated():
    from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch

    get_or_fetch("backend/a", ttl_seconds=60, fetch_fn=lambda: {"name": "a"})
    result = get_or_fetch("backend/b", ttl_seconds=60, fetch_fn=lambda: {"name": "b"})

    assert result.payload == {"name": "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_cache.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_cache.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_cache.py
"""File-based TTL cache for IBM Quantum data. New pattern for this repo -- no
existing cache to model here, since every other tool in chatqec_mcp_tools is a
pure, stateless computation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class CacheResult:
    payload: Any
    fetched_at: float
    age_seconds: float
    stale: bool = False


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    root = Path(base) / "chatqec-mcp-tools" / "ibm_hardware"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "__")
    return _cache_root() / f"{safe}.json"


def get_or_fetch(key: str, ttl_seconds: float, fetch_fn: Callable[[], Any]) -> CacheResult:
    """Return cached payload if fresh; otherwise call fetch_fn and persist the
    result. If fetch_fn raises and a (possibly expired) cached payload exists,
    serve it with stale=True instead of failing. If fetch_fn raises and there is
    no cached payload at all, the exception propagates."""
    path = _cache_path(key)
    now = time.time()
    cached_raw: dict[str, Any] | None = None

    if path.exists():
        try:
            cached_raw = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = float(cached_raw["fetched_at"])
            age = now - fetched_at
            if age < ttl_seconds:
                return CacheResult(payload=cached_raw["payload"], fetched_at=fetched_at, age_seconds=age)
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            cached_raw = None

    try:
        payload = fetch_fn()
    except Exception:
        if cached_raw is not None:
            fetched_at = float(cached_raw["fetched_at"])
            return CacheResult(
                payload=cached_raw["payload"],
                fetched_at=fetched_at,
                age_seconds=now - fetched_at,
                stale=True,
            )
        raise

    fetched_at = time.time()
    path.write_text(
        json.dumps({"fetched_at": fetched_at, "ttl_seconds": ttl_seconds, "payload": payload}),
        encoding="utf-8",
    )
    return CacheResult(payload=payload, fetched_at=fetched_at, age_seconds=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_cache.py -v`
Expected: `PASS` (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_cache.py tests/test_ibm_cache.py
git commit -m "feat(ibm): add file-based TTL cache for IBM Quantum data"
```

---

### Task 3: `ibm_list_backends` tool

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/list_backends.py`
- Modify: `src/chatqec_mcp_tools/server.py`
- Test: `tests/test_ibm_list_backends.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `get_ibm_service()` / `IBMCredentialsError` from Task 1's `_client.py`;
  `get_or_fetch(key, ttl_seconds, fetch_fn) -> CacheResult` from Task 2's `_cache.py`.
- Produces: `BackendInfo`, `IbmListBackendsInput`, `IbmListBackendsOutput`,
  `ibm_list_backends(inp: IbmListBackendsInput) -> IbmListBackendsOutput`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_list_backends.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_list_backends.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `list_backends.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/list_backends.py
"""ibm_list_backends tool -- backends visible to the configured IBM Quantum account."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service

_DEFAULT_TTL_SECONDS = 60.0


class BackendInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    num_qubits: int
    operational: bool
    pending_jobs: int
    basis_gates: list[str]


class IbmListBackendsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IbmListBackendsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backends: list[BackendInfo] = []
    error_message: str = ""


def ibm_list_backends(inp: IbmListBackendsInput) -> IbmListBackendsOutput:
    ttl = float(os.environ.get("CHATQEC_IBM_BACKEND_LIST_TTL_SECONDS", _DEFAULT_TTL_SECONDS))

    def fetch() -> list[dict]:
        service = get_ibm_service()
        result = []
        for backend in service.backends():
            status = backend.status()
            result.append(
                {
                    "name": backend.name,
                    "num_qubits": backend.num_qubits,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "basis_gates": list(backend.operation_names),
                }
            )
        return result

    try:
        cache_result = get_or_fetch("list_backends", ttl, fetch)
    except IBMCredentialsError as exc:
        return IbmListBackendsOutput(ok=False, error_message=str(exc))
    except Exception as exc:  # network/SDK errors, never raise out of the tool
        return IbmListBackendsOutput(ok=False, error_message=f"Failed to list backends: {exc}")

    return IbmListBackendsOutput(
        ok=True, backends=[BackendInfo.model_validate(b) for b in cache_result.payload]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_list_backends.py -v`
Expected: `PASS` (2 tests)

- [ ] **Step 5: Register the tool in `server.py`**

Read `src/chatqec_mcp_tools/server.py` first to find the existing `_TOOLS` list and an
existing handler (e.g. `_code_params_handler`) to match the exact style, then add, near the
other handlers:

```python
from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
    IbmListBackendsInput,
    ibm_list_backends,
)


def _ibm_list_backends_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_list_backends(IbmListBackendsInput.model_validate(args)).model_dump()
```

and add to the `_TOOLS` list:

```python
    ToolSpec(
        name="ibm_list_backends",
        description=(
            "List IBM Quantum backends visible to the configured account: name, "
            "qubit count, operational status, and pending job queue depth. "
            "Free, cached for 60 seconds."
        ),
        input_schema=IbmListBackendsInput.model_json_schema(),
        handler=_ibm_list_backends_handler,
    ),
```

- [ ] **Step 6: Add a registration test to `test_server.py`**

Follow the existing pattern in that file (one assertion block per tool) and add:

```python
def test_ibm_list_backends_is_registered():
    server = build_server()
    assert "ibm_list_backends" in {t.name for t in server.list_tools()}
```

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests/test_ibm_list_backends.py tests/test_server.py -v`
Expected: all `PASS`

- [ ] **Step 8: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/list_backends.py \
        src/chatqec_mcp_tools/server.py \
        tests/test_ibm_list_backends.py tests/test_server.py
git commit -m "feat(ibm): add ibm_list_backends tool"
```

---

### Task 4: `_hardware_graph.py` — live backend → normalized graph

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_hardware_graph.py`
- Test: `tests/test_ibm_hardware_graph.py`

**Interfaces:**
- Consumes: nothing from prior tasks (takes a backend-like object directly, so it's testable
  without `_client.py`).
- Produces: `HardwareGraph` (dataclass: `graph: networkx.Graph`, `backend_name: str`,
  `.num_qubits` property), `build_hardware_graph(backend: Any) -> HardwareGraph`. Consumed by
  Tasks 6 and 8/9 (placement) and 10 (diagram). Graph node attrs: `T1`, `T2`,
  `readout_error`, `readout_fidelity`, `is_operational` (all `float | None` except the bool).
  Graph edge attrs: `native_gate`, `gate_error`, `gate_time`, `gate_fidelity` (all `str |
  float | None`).

**Note on source lineage:** this adapts (not copies verbatim) `cahgr/hardware_kg.py` — same
schema and the same native-two-qubit-gate priority list (already correctly anticipates a
coupler exposing more than one candidate gate, e.g. Heron's `cz`+`rzz`, confirmed live
2026-09-03), but reads a live `qiskit_ibm_runtime` backend object instead of a parsed JSON
snapshot, and drops the intern's separate `hardware_layout` coordinate-lookup dependency.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_hardware_graph.py
from __future__ import annotations

from types import SimpleNamespace


class _FakeProperties:
    """Mimics qiskit's BackendProperties interface for the two calls
    build_hardware_graph needs: qubit_property(q) and gate_error/gate_length."""

    def __init__(self, qubits: dict[int, dict], gates: dict[tuple, dict]):
        self._qubits = qubits
        self._gates = gates  # {(gate_name, (u, v)): {"error": ..., "length": ...}}
        self.last_update_date = "2026-09-03T14:14:15-04:00"

    def qubit_property(self, q: int) -> dict:
        return self._qubits.get(q, {})

    def gate_error(self, gate: str, qubits) -> float:
        key = (gate, tuple(qubits))
        if key not in self._gates:
            raise ValueError(f"no calibration for {gate} on {qubits}")
        return self._gates[key]["error"]

    def gate_length(self, gate: str, qubits) -> float:
        key = (gate, tuple(qubits))
        return self._gates[key]["length"]


def _fake_backend():
    props = _FakeProperties(
        qubits={
            0: {"T1": (3.0e-05, "s"), "T2": (2.7e-05, "s"), "readout_error": (0.0127, "")},
            1: {"T1": (2.5e-05, "s"), "T2": (2.0e-05, "s"), "readout_error": (0.02, "")},
        },
        gates={("cz", (0, 1)): {"error": 0.0097, "length": 6.8e-08}},
    )
    return SimpleNamespace(
        name="ibm_fake",
        num_qubits=2,
        coupling_map=[(0, 1), (1, 0)],
        properties=lambda: props,
    )


def test_build_hardware_graph_populates_node_attrs():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    assert hg.backend_name == "ibm_fake"
    assert hg.num_qubits == 2
    q0 = hg.graph.nodes[0]
    assert q0["T1"] == 3.0e-05
    assert q0["readout_fidelity"] == pytest_approx(1 - 0.0127)
    assert q0["is_operational"] is True


def test_build_hardware_graph_picks_native_gate_by_priority():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    edge = hg.graph.edges[0, 1]
    assert edge["native_gate"] == "cz"
    assert edge["gate_error"] == 0.0097
    assert edge["gate_fidelity"] == pytest_approx(1 - 0.0097)


def test_build_hardware_graph_deduplicates_bidirectional_coupling_pairs():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    hg = build_hardware_graph(_fake_backend())

    assert hg.graph.number_of_edges() == 1  # (0,1) and (1,0) collapse to one undirected edge


def test_is_operational_false_when_t1_missing():
    from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

    props = _FakeProperties(
        qubits={0: {"readout_error": (0.01, "")}, 1: {"T1": (2.5e-05, "s"), "T2": (2.0e-05, "s")}},
        gates={},
    )
    backend = SimpleNamespace(
        name="ibm_fake2", num_qubits=2, coupling_map=[(0, 1)], properties=lambda: props
    )

    hg = build_hardware_graph(backend)

    assert hg.graph.nodes[0]["is_operational"] is False


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_hardware_graph.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_hardware_graph.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_hardware_graph.py
"""Live IBM backend -> normalized property graph.

Adapted from cahgr/hardware_kg.py in the intern's
HardwareAwarenessIntegrationPipeline: same node/edge schema and the same
native-two-qubit-gate priority list, but built from a live qiskit_ibm_runtime
backend's .properties()/.coupling_map instead of a parsed JSON snapshot. The
intern's separate hardware_layout.vendor_ibm (JSON reconstruction) and
hardware_layout.qubit_coordinates (per-chip coordinate table) are intentionally
not ported -- unnecessary once the source is a live SDK object, and the
coupling-map diagram (see _coupling_diagram.py) computes its own layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

# Priority order to look for on a coupler. Different chip families report
# different native gates (Heron: cz + rzz; Eagle/older chips: cx or ecr) --
# confirmed live 2026-09-03 that a single coupler can expose more than one
# (ibm_fez qubits (0,1): both cz and rzz). Picking by priority rather than
# tracking every alternative is a deliberate simplification for scoring.
_TWO_Q_GATE_PRIORITY = ("cz", "ecr", "cx", "rzz", "iswap", "swap")


@dataclass
class HardwareGraph:
    graph: nx.Graph
    backend_name: str

    @property
    def num_qubits(self) -> int:
        return self.graph.number_of_nodes()


def build_hardware_graph(backend: Any) -> HardwareGraph:
    """Build the normalized hardware graph from a live backend object (anything
    with .name, .num_qubits, .coupling_map, .properties())."""
    props = backend.properties()

    graph = nx.Graph()
    graph.graph["backend_name"] = backend.name
    graph.graph["last_update_date"] = str(getattr(props, "last_update_date", None))

    for q in range(backend.num_qubits):
        qprop = props.qubit_property(q) if props is not None else {}
        t1 = _first(qprop.get("T1"))
        t2 = _first(qprop.get("T2"))
        readout_error = _first(qprop.get("readout_error"))
        graph.add_node(
            q,
            T1=t1,
            T2=t2,
            readout_error=readout_error,
            readout_fidelity=_fidelity_from_error(readout_error),
            is_operational=_is_operational(t1, t2, readout_error),
        )

    for u, v in backend.coupling_map:
        if graph.has_edge(u, v):
            continue
        metrics = _pick_native_two_qubit_metrics(props, u, v) if props is not None else _empty_metrics()
        graph.add_edge(u, v, **metrics)

    return HardwareGraph(graph=graph, backend_name=backend.name)


def _first(value: Any) -> float | None:
    """qiskit's qubit_property() values are (value, unit) tuples; unwrap."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _pick_native_two_qubit_metrics(props: Any, u: int, v: int) -> dict[str, Any]:
    for gate in _TWO_Q_GATE_PRIORITY:
        for pair in ((u, v), (v, u)):
            try:
                error = props.gate_error(gate, pair)
                duration = props.gate_length(gate, pair)
            except Exception:
                continue
            return {
                "native_gate": gate,
                "gate_error": error,
                "gate_time": duration,
                "gate_fidelity": _fidelity_from_error(error),
            }
    return _empty_metrics()


def _empty_metrics() -> dict[str, Any]:
    return {"native_gate": None, "gate_error": None, "gate_time": None, "gate_fidelity": None}


def _fidelity_from_error(error: Any) -> float | None:
    if error is None:
        return None
    try:
        return max(0.0, 1.0 - float(error))
    except (TypeError, ValueError):
        return None


def _is_operational(t1: Any, t2: Any, readout_error: Any) -> bool:
    for value in (t1, t2):
        if value is None:
            return False
        try:
            if math.isnan(float(value)) or float(value) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    if readout_error is not None:
        try:
            if float(readout_error) >= 0.5:
                return False
        except (TypeError, ValueError):
            pass
    return True


__all__ = ["HardwareGraph", "build_hardware_graph"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_hardware_graph.py -v`
Expected: `PASS` (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_hardware_graph.py tests/test_ibm_hardware_graph.py
git commit -m "feat(ibm): build normalized hardware graph from a live IBM backend"
```

---

### Task 5: `ibm_backend_calibration` tool

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/calibration.py`
- Modify: `src/chatqec_mcp_tools/server.py`
- Test: `tests/test_ibm_calibration.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `get_ibm_service`/`IBMCredentialsError` (Task 1), `get_or_fetch`/`CacheResult`
  (Task 2), `build_hardware_graph`/`HardwareGraph` (Task 4).
- Produces: `QubitCalibration`, `CouplerCalibration`, `IbmBackendCalibrationInput`,
  `IbmBackendCalibrationOutput` (fields: `ok, backend, fetched_at: str (ISO8601),
  age_seconds: int, stale: bool, qubits: list[QubitCalibration], couplers:
  list[CouplerCalibration], error_message`), `ibm_backend_calibration(inp) -> Output`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_calibration.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _fake_service_with_graph_payload():
    # calibration.py serializes HardwareGraph to a plain dict before caching;
    # simulate that dict shape directly so this test doesn't depend on Task 4's
    # internals -- it patches build_hardware_graph, not the fake backend.
    class _Graph:
        def __init__(self):
            import networkx as nx

            g = nx.Graph()
            g.add_node(0, T1=3.0e-05, T2=2.7e-05, readout_error=0.0127, readout_fidelity=0.9873, is_operational=True)
            g.add_node(1, T1=2.5e-05, T2=2.0e-05, readout_error=0.02, readout_fidelity=0.98, is_operational=True)
            g.add_edge(0, 1, native_gate="cz", gate_error=0.0097, gate_time=6.8e-08, gate_fidelity=0.9903)
            self.graph = g
            self.backend_name = "ibm_fake"

    service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))
    return service, _Graph()


def test_ibm_backend_calibration_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    service, hw_graph = _fake_service_with_graph_payload()

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service", return_value=service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.calibration.build_hardware_graph", return_value=hw_graph),
    ):
        out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is True
    assert out.backend == "ibm_fake"
    assert out.age_seconds == 0
    assert out.stale is False
    assert {q.index for q in out.qubits} == {0, 1}
    q0 = next(q for q in out.qubits if q.index == 0)
    assert q0.t1_us == pytest_approx(30.0)  # seconds -> microseconds
    assert len(out.couplers) == 1
    assert out.couplers[0].native_gate == "cz"


def test_ibm_backend_calibration_second_call_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    service, hw_graph = _fake_service_with_graph_payload()
    call_count = {"n": 0}

    def build_and_count(backend):
        call_count["n"] += 1
        return hw_graph

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service", return_value=service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.calibration.build_hardware_graph", side_effect=build_and_count),
    ):
        ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))
        out2 = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert call_count["n"] == 1
    assert out2.age_seconds >= 0


def test_ibm_backend_calibration_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )

    with patch(
        "chatqec_mcp_tools.tools.ibm_hardware.calibration.get_ibm_service",
        side_effect=IBMCredentialsError("no token configured"),
    ):
        out = ibm_backend_calibration(IbmBackendCalibrationInput(backend="ibm_fake"))

    assert out.ok is False
    assert "no token configured" in out.error_message


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_calibration.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `calibration.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/calibration.py
"""ibm_backend_calibration tool -- T1/T2/readout/gate-error for one backend,
TTL-cached, always reports its own data age."""

from __future__ import annotations

import datetime as dt
import os

from pydantic import BaseModel, ConfigDict

from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

_DEFAULT_TTL_SECONDS = 3600.0


class QubitCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    t1_us: float | None
    t2_us: float | None
    readout_error: float | None


class CouplerCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubits: list[int]
    native_gate: str | None
    two_qubit_error: float | None
    gate_duration_ns: float | None


class IbmBackendCalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str


class IbmBackendCalibrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    stale: bool = False
    qubits: list[QubitCalibration] = []
    couplers: list[CouplerCalibration] = []
    error_message: str = ""


def ibm_backend_calibration(inp: IbmBackendCalibrationInput) -> IbmBackendCalibrationOutput:
    ttl = float(os.environ.get("CHATQEC_IBM_CALIBRATION_TTL_SECONDS", _DEFAULT_TTL_SECONDS))

    def fetch() -> dict:
        service = get_ibm_service()
        backend = service.backend(inp.backend)
        hw_graph = build_hardware_graph(backend)
        qubits = [
            {
                "index": q,
                "t1_us": (d["T1"] * 1e6) if d.get("T1") is not None else None,
                "t2_us": (d["T2"] * 1e6) if d.get("T2") is not None else None,
                "readout_error": d.get("readout_error"),
            }
            for q, d in hw_graph.graph.nodes(data=True)
        ]
        couplers = [
            {
                "qubits": [u, v],
                "native_gate": d.get("native_gate"),
                "two_qubit_error": d.get("gate_error"),
                "gate_duration_ns": (d["gate_time"] * 1e9) if d.get("gate_time") is not None else None,
            }
            for u, v, d in hw_graph.graph.edges(data=True)
        ]
        return {"qubits": qubits, "couplers": couplers}

    try:
        cache_result = get_or_fetch(f"calibration/{inp.backend}", ttl, fetch)
    except IBMCredentialsError as exc:
        return IbmBackendCalibrationOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IbmBackendCalibrationOutput(
            ok=False, error_message=f"Failed to fetch calibration for {inp.backend}: {exc}"
        )

    fetched_at_iso = dt.datetime.fromtimestamp(cache_result.fetched_at, tz=dt.UTC).isoformat()
    return IbmBackendCalibrationOutput(
        ok=True,
        backend=inp.backend,
        fetched_at=fetched_at_iso,
        age_seconds=int(cache_result.age_seconds),
        stale=cache_result.stale,
        qubits=[QubitCalibration.model_validate(q) for q in cache_result.payload["qubits"]],
        couplers=[CouplerCalibration.model_validate(c) for c in cache_result.payload["couplers"]],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_calibration.py -v`
Expected: `PASS` (3 tests)

- [ ] **Step 5: Register in `server.py`** (same pattern as Task 3 Step 5 — handler +
`ToolSpec` with description: *"Fetch T1/T2/readout error and per-coupler two-qubit gate
error/duration for one IBM backend. Cached; every answer built from this must state the
data's age (e.g. 'fetched 12 min ago')."*)

- [ ] **Step 6: Add registration test to `test_server.py`** (same pattern as Task 3 Step 6,
name `ibm_backend_calibration`)

- [ ] **Step 7: Run full suite, then commit**

```bash
pytest tests/test_ibm_calibration.py tests/test_server.py -v
git add src/chatqec_mcp_tools/tools/ibm_hardware/calibration.py \
        src/chatqec_mcp_tools/server.py \
        tests/test_ibm_calibration.py tests/test_server.py
git commit -m "feat(ibm): add ibm_backend_calibration tool"
```

---

### Task 6: `_quality_model.py` — objective-based scoring

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_quality_model.py`
- Test: `tests/test_ibm_quality_model.py`

**Interfaces:**
- Consumes: `HardwareGraph` (Task 4) — specifically its `.graph` attribute (a
  `networkx.Graph` with the node/edge attrs Task 4 defines).
- Produces: `HardwareQualityModel` class with `__init__(hw_graph, *, objective: str =
  "balanced")`, `.node_score(q: int) -> float`, `.edge_score(u: int, v: int) -> float`.
  Consumed by Task 8 (`_placement.py`).

**Source lineage:** ported essentially as-is from `cahgr/retrieval.py`'s
`HardwareQualityModel`, adjusted only to read from `HardwareGraph` instead of
`HardwareKnowledgeGraph` (same attribute names, so the scoring logic is unchanged) and to
drop the `reliability_adjustment` feedback-loop parameter (that's part of the intern's
Step 8 feedback loop, out of scope here).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_quality_model.py
from __future__ import annotations

import networkx as nx


def _build_graph():
    g = nx.Graph()
    g.add_node(0, T1=30.0, T2=27.0, readout_fidelity=0.99, is_operational=True)
    g.add_node(1, T1=10.0, T2=8.0, readout_fidelity=0.95, is_operational=True)
    g.add_node(2, T1=None, T2=None, readout_fidelity=None, is_operational=False)
    g.add_edge(0, 1, gate_fidelity=0.99, gate_time=68.0)
    g.add_edge(1, 2, gate_fidelity=0.90, gate_time=200.0)
    return g


def test_node_score_prefers_higher_t1_t2_readout_by_default():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.node_score(0) > model.node_score(1)


def test_inoperational_qubit_scores_zero():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.node_score(2) == 0.0


def test_maximum_fidelity_objective_uses_readout_only():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="maximum_fidelity")

    # qubit 0 has both better T1/T2 AND better readout, so this doesn't
    # distinguish "readout-only" from "averaged" on its own -- assert the
    # concrete value instead of just an ordering, since maximum_fidelity should
    # equal the min-max normalized readout_fidelity exactly (0.95 -> 0, 0.99 -> 1).
    assert model.node_score(0) == 1.0
    assert model.node_score(1) == 0.0


def test_edge_score_uses_gate_fidelity_by_default():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.edge_score(0, 1) > model.edge_score(1, 2)


def test_minimum_execution_time_objective_favors_faster_gate():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="minimum_execution_time")

    assert model.edge_score(0, 1) > model.edge_score(1, 2)  # 68ns beats 200ns


def test_edge_score_zero_for_nonexistent_edge():
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    model = HardwareQualityModel(_build_graph(), objective="balanced")

    assert model.edge_score(0, 2) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_quality_model.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_quality_model.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_quality_model.py
"""Objective-based, min-max normalized hardware quality scoring.

Ported essentially as-is from cahgr/retrieval.py's HardwareQualityModel in the
intern's HardwareAwarenessIntegrationPipeline -- the review found this schema
sound. Adjusted to read from this package's HardwareGraph (same attribute
names) and drop the reliability_adjustment feedback-loop parameter (out of
scope here).
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import networkx as nx


class HardwareQualityModel:
    def __init__(self, graph: nx.Graph, *, objective: str = "balanced") -> None:
        self.graph = graph
        self.objective = objective

        t1_vals = _collect(graph.nodes(data=True), "T1")
        t2_vals = _collect(graph.nodes(data=True), "T2")
        ro_vals = _collect(graph.nodes(data=True), "readout_fidelity")
        fid_vals = [d["gate_fidelity"] for _, _, d in graph.edges(data=True) if d.get("gate_fidelity") is not None]
        time_vals = [d["gate_time"] for _, _, d in graph.edges(data=True) if d.get("gate_time") is not None]

        self._t1_norm = _minmax(t1_vals)
        self._t2_norm = _minmax(t2_vals)
        self._ro_norm = _minmax(ro_vals)
        self._fid_norm = _minmax(fid_vals)
        self._time_norm = _minmax(time_vals)

    def node_score(self, q: int) -> float:
        d = self.graph.nodes[q]
        if not d.get("is_operational", True):
            return 0.0

        if self.objective == "maximum_fidelity":
            return self._ro_norm(d["readout_fidelity"]) if d.get("readout_fidelity") is not None else 0.5

        parts = []
        if d.get("T1") is not None:
            parts.append(self._t1_norm(d["T1"]))
        if d.get("T2") is not None:
            parts.append(self._t2_norm(d["T2"]))
        if d.get("readout_fidelity") is not None:
            parts.append(self._ro_norm(d["readout_fidelity"]))
        return mean(parts) if parts else 0.5

    def edge_score(self, u: int, v: int) -> float:
        if not self.graph.has_edge(u, v):
            return 0.0
        d = self.graph.edges[u, v]

        if self.objective == "minimum_execution_time":
            if d.get("gate_time") is None:
                return 0.5
            return 1.0 - self._time_norm(d["gate_time"])

        if d.get("gate_fidelity") is None:
            return 0.5
        return self._fid_norm(d["gate_fidelity"])


def _collect(items: Any, key: str) -> list[float]:
    values = []
    for _, data in items:
        v = data.get(key)
        if v is not None:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass
    return values


def _minmax(values: list[float]):
    if not values:
        return lambda v: 0.5
    lo, hi = min(values), max(values)
    if hi <= lo:
        return lambda v: 1.0

    def _norm(v: float) -> float:
        try:
            return (float(v) - lo) / (hi - lo)
        except (TypeError, ValueError):
            return 0.5

    return _norm


__all__ = ["HardwareQualityModel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_quality_model.py -v`
Expected: `PASS` (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_quality_model.py tests/test_ibm_quality_model.py
git commit -m "feat(ibm): port objective-based hardware quality scoring"
```

---

### Task 7: `_circuits.py` — QASM parsing, presets, interaction graph

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_circuits.py`
- Test: `tests/test_ibm_circuits.py`

**Interfaces:**
- Consumes: nothing from prior tasks (lazily imports `qiskit`).
- Produces: `CircuitSpec` (dataclass: `circuit: Any` [a `QuantumCircuit`],
  `num_logical_qubits: int`), `parse_circuit_input(*, circuit_qasm: str | None,
  circuit_preset: str | None, num_qubits: int | None, qaoa_layers: int | None) ->
  CircuitSpec` (raises `ValueError` on bad input — caller wraps in try/except),
  `InteractionGraph` (dataclass: `graph: networkx.Graph`), `build_interaction_graph(circuit:
  Any) -> InteractionGraph`. Consumed by Task 9 (`_placement.py` via the interaction graph)
  and Task 11 (`map_circuit.py`).

**Source lineage:** `build_interaction_graph` adapts `cahgr/circuit_graph.py`'s
`build_circuit_interaction_graph` (verified 2026-09-03: it already dispatches on a real
`QuantumCircuit`, so this is a near-verbatim port of the Qiskit-circuit code path only — the
intern's dict-instruction-list fallback path is dropped since this package always has qiskit
available once QASM parsing needs it). Presets (`bell`, `ghz`, `qft`, `qaoa`) adapt the
correct, straightforward constructions in `qec/prompt_library.py`'s `_ghz`/`_qft`/`_qaoa`
(verified correct: standard textbook GHZ/QFT/ring-MaxCut-QAOA — no defects found).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_circuits.py
from __future__ import annotations

import pytest


def test_parse_circuit_input_from_qasm():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    qasm = (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n"
        "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
    )
    spec = parse_circuit_input(circuit_qasm=qasm, circuit_preset=None, num_qubits=None, qaoa_layers=None)

    assert spec.num_logical_qubits == 2


def test_parse_circuit_input_rejects_both_qasm_and_preset():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(circuit_qasm="OPENQASM 2.0;", circuit_preset="bell", num_qubits=None, qaoa_layers=None)


def test_parse_circuit_input_rejects_neither():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(circuit_qasm=None, circuit_preset=None, num_qubits=None, qaoa_layers=None)


def test_bell_preset_is_two_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="bell", num_qubits=None, qaoa_layers=None)

    assert spec.num_logical_qubits == 2
    assert spec.circuit.count_ops().get("cx", 0) == 1


def test_ghz_preset_respects_num_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="ghz", num_qubits=5, qaoa_layers=None)

    assert spec.num_logical_qubits == 5
    assert spec.circuit.count_ops().get("cx", 0) == 4


def test_qaoa_preset_layers_multiply_two_qubit_gate_count():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    one_layer = parse_circuit_input(circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=1)
    two_layers = parse_circuit_input(circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=2)

    assert two_layers.circuit.count_ops().get("cx", 0) == 2 * one_layer.circuit.count_ops().get("cx", 0)


def test_build_interaction_graph_weights_repeated_pairs():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
        build_interaction_graph,
        parse_circuit_input,
    )

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="ghz", num_qubits=3, qaoa_layers=None)
    interaction = build_interaction_graph(spec.circuit)

    assert set(interaction.graph.nodes) == {0, 1, 2}
    assert interaction.graph.has_edge(0, 1)
    assert interaction.graph.has_edge(0, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_circuits.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_circuits.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_circuits.py
"""Circuit input handling: QASM parsing, named presets, and the two-qubit
interaction graph the placement engine matches against hardware.

build_interaction_graph adapts cahgr/circuit_graph.py's
build_circuit_interaction_graph (verified 2026-09-03: it already dispatches on
a real Qiskit QuantumCircuit, so only that code path is ported here -- the
generalization to arbitrary circuits this package needs was already present in
the intern's code, not something added). Presets adapt qec/prompt_library.py's
_ghz/_qft/_qaoa (verified correct: standard GHZ/QFT/ring-MaxCut-QAOA
constructions).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx

_PRESETS = ("bell", "ghz", "qft", "qaoa")


@dataclass
class CircuitSpec:
    circuit: Any  # a qiskit.circuit.QuantumCircuit
    num_logical_qubits: int


@dataclass
class InteractionGraph:
    graph: nx.Graph


def parse_circuit_input(
    *,
    circuit_qasm: str | None,
    circuit_preset: str | None,
    num_qubits: int | None,
    qaoa_layers: int | None,
) -> CircuitSpec:
    if (circuit_qasm is None) == (circuit_preset is None):
        raise ValueError("Provide exactly one of circuit_qasm or circuit_preset.")

    from qiskit import QuantumCircuit  # noqa: PLC0415

    if circuit_qasm is not None:
        try:
            circuit = QuantumCircuit.from_qasm_str(circuit_qasm)
        except Exception as exc:
            raise ValueError(f"Could not parse circuit_qasm: {exc}") from exc
        return CircuitSpec(circuit=circuit, num_logical_qubits=circuit.num_qubits)

    if circuit_preset not in _PRESETS:
        raise ValueError(f"circuit_preset must be one of {_PRESETS}, got {circuit_preset!r}")

    n = num_qubits or (2 if circuit_preset == "bell" else 3)
    if circuit_preset == "bell":
        if n != 2:
            raise ValueError("the bell preset is defined for exactly 2 qubits")
        circuit = _bell()
    elif circuit_preset == "ghz":
        circuit = _ghz(n)
    elif circuit_preset == "qft":
        circuit = _qft(n)
    else:
        circuit = _qaoa(n, layers=qaoa_layers or 1)

    return CircuitSpec(circuit=circuit, num_logical_qubits=n)


def build_interaction_graph(circuit: Any) -> InteractionGraph:
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    qubit_index = {qubit: i for i, qubit in enumerate(circuit.qubits)}

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure", "reset", "delay"}:
            continue
        idxs = [qubit_index[q] for q in instruction.qubits]
        if len(idxs) == 2:
            u, v = idxs
            if graph.has_edge(u, v):
                graph.edges[u, v]["weight"] += 1
            else:
                graph.add_edge(u, v, weight=1)

    return InteractionGraph(graph=graph)


def _bell():
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _ghz(num_qubits: int):
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"ghz{num_qubits}")
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    return qc


def _qft(num_qubits: int):
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"qft{num_qubits}")
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            qc.cp(math.pi / 2 ** (k - j), k, j)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    return qc


def _qaoa(num_qubits: int, *, layers: int):
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"qaoa_maxcut_ring{num_qubits}_p{layers}")
    gamma, beta = math.pi / 4, math.pi / 8
    for _ in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            j = (i + 1) % num_qubits
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(num_qubits):
            qc.rx(2 * beta, i)
    return qc


__all__ = ["CircuitSpec", "InteractionGraph", "parse_circuit_input", "build_interaction_graph"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_circuits.py -v`
Expected: `PASS` (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_circuits.py tests/test_ibm_circuits.py
git commit -m "feat(ibm): add QASM parsing, circuit presets, and interaction graph builder"
```

---

### Task 8: `_placement.py` — greedy region-growing

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py`
- Test: `tests/test_ibm_placement.py`

**Interfaces:**
- Consumes: `HardwareQualityModel` (Task 6), `InteractionGraph` (Task 7).
- Produces: `PlacementResult` (dataclass: `assignment: dict[int, int]` [logical → physical],
  `strategy_used: str`), `place_greedy(hw_graph: networkx.Graph, interaction_graph:
  networkx.Graph, quality: HardwareQualityModel, *, connectivity_weight: float = 0.7,
  calibration_weight: float = 0.3) -> PlacementResult`. Consumed by Task 9's `place()`
  dispatcher and Task 11 (`map_circuit.py`).

**Source lineage:** ported essentially as-is from `cahgr/retrieval.py`'s `_retrieve_greedy` /
`_physical_candidates` — no defects found in this strategy during review.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_placement.py
from __future__ import annotations

import networkx as nx


def _line_hardware_graph(n: int):
    """0-1-2-3-...-n-1 line, uniform quality except qubit 0 is best."""
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=100.0 if q == 0 else 50.0, T2=90.0 if q == 0 else 45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=50.0)
    return g


def test_place_greedy_returns_valid_mapping_for_ghz():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)  # GHZ-3 star: logical 0 talks to 1 and 2

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert result.strategy_used == "greedy"
    assert set(result.assignment.keys()) == {0, 1, 2}
    assert len(set(result.assignment.values())) == 3  # no physical qubit reused


def test_place_greedy_seeds_on_highest_quality_qubit():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=5)  # logical 0 is the most-interacting node

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert result.assignment[0] == 0  # physical qubit 0 is the best-quality seed


def test_place_greedy_handles_isolated_logical_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)
    interaction = nx.Graph()
    interaction.add_nodes_from([0, 1, 2])  # no edges -- e.g. three idle spectator qubits

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_greedy(hw_graph, interaction, quality)

    assert len(result.assignment) == 3
    assert len(set(result.assignment.values())) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_placement.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_placement.py`** (greedy strategy only — VF2 added in Task 9)

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py
"""Circuit-to-hardware placement strategies.

place_greedy is ported essentially as-is from cahgr/retrieval.py's
_retrieve_greedy/_physical_candidates -- no defects found there during review.
place_vf2 (added in the next task) fixes the one confirmed real defect: the
intern's hardcoded _VF2_NODE_CAP/_VF2_CANDIDATE_CAP had no wall-clock bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel


@dataclass
class PlacementResult:
    assignment: dict[int, int]
    strategy_used: str


def place_greedy(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    quality: HardwareQualityModel,
    *,
    connectivity_weight: float = 0.7,
    calibration_weight: float = 0.3,
) -> PlacementResult:
    strength = {
        n: sum(d["weight"] for _, _, d in interaction_graph.edges(n, data=True))
        for n in interaction_graph.nodes
    }
    order = sorted(interaction_graph.nodes, key=lambda n: -strength.get(n, 0))

    assignment: dict[int, int] = {}
    used_physical: set[int] = set()

    if order:
        first = order[0]
        seed = max(hw_graph.nodes, key=lambda q: quality.node_score(q))
        assignment[first] = seed
        used_physical.add(seed)

        for logical in order[1:]:
            candidates = _physical_candidates(hw_graph, interaction_graph, logical, assignment, used_physical)
            if not candidates:
                continue

            def score(phys: int, logical=logical) -> float:
                s = calibration_weight * quality.node_score(phys)
                conn = 0.0
                for neighbor in interaction_graph.neighbors(logical):
                    if neighbor in assignment:
                        partner = assignment[neighbor]
                        w = interaction_graph.edges[logical, neighbor]["weight"]
                        if hw_graph.has_edge(phys, partner):
                            conn += w * quality.edge_score(phys, partner)
                return s + connectivity_weight * conn

            best = max(candidates, key=score)
            assignment[logical] = best
            used_physical.add(best)

    remaining_logical = [n for n in interaction_graph.nodes if n not in assignment]
    remaining_physical = sorted(set(hw_graph.nodes) - used_physical, key=lambda q: -quality.node_score(q))
    for logical, phys in zip(remaining_logical, remaining_physical):
        assignment[logical] = phys
        used_physical.add(phys)

    if len(assignment) < interaction_graph.number_of_nodes():
        missing = interaction_graph.number_of_nodes() - len(assignment)
        raise ValueError(
            f"Could not place {missing} logical qubit(s): the hardware graph "
            f"({hw_graph.number_of_nodes()} qubits) is smaller than required."
        )

    return PlacementResult(assignment=assignment, strategy_used="greedy")


def _physical_candidates(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    logical_node: int,
    assignment: dict[int, int],
    used_physical: set[int],
) -> set[int]:
    candidates: set[int] = set()
    for neighbor in interaction_graph.neighbors(logical_node):
        if neighbor in assignment:
            phys = assignment[neighbor]
            candidates.update(set(hw_graph.neighbors(phys)) - used_physical)
    if not candidates:
        candidates = set(hw_graph.nodes) - used_physical
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_placement.py -v`
Expected: `PASS` (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py tests/test_ibm_placement.py
git commit -m "feat(ibm): add greedy region-growing placement strategy"
```

---

### Task 9: `_placement.py` — time-budgeted VF2 + `place()` dispatcher

**Files:**
- Modify: `src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py`
- Modify: `tests/test_ibm_placement.py`

**Interfaces:**
- Consumes: same as Task 8, plus `networkx.algorithms.isomorphism.GraphMatcher`.
- Produces: `place_vf2(hw_graph, interaction_graph, quality, *, time_budget_seconds: float =
  2.0) -> PlacementResult | None` (returns `None` on no match/budget exhausted, never
  raises), `place(hw_graph, interaction_graph, quality, *, connectivity_weight=0.7,
  calibration_weight=0.3) -> PlacementResult` (tries VF2 first, falls back to greedy).
  Consumed by Task 11 (`map_circuit.py`) and Task 12/13 (preview/submit, indirectly through
  `map_circuit`'s helper).

**This is the task that fixes the confirmed real defect** in `cahgr/retrieval.py:47-48`:
`_VF2_NODE_CAP = 12` / `_VF2_CANDIDATE_CAP = 4000` are hardcoded constants with no complexity
justification and — the actual gap — no wall-clock bound, so a pathological interaction-graph
shape within the node cap could still spend a long time in VF2's internal backtracking before
yielding its first candidate. This port replaces both constants with a time budget checked on
every loop iteration.

- [ ] **Step 1: Add the failing tests to `tests/test_ibm_placement.py`**

```python
# append to tests/test_ibm_placement.py

def test_place_vf2_finds_exact_match_when_one_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)  # 0-1-2-3-4
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(1, 2, weight=1)  # a 3-node path -- exists as a subgraph of the line

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality)

    assert result is not None
    assert result.strategy_used == "vf2"
    pu, pv = result.assignment[0], result.assignment[1]
    pv2 = result.assignment[2]
    assert hw_graph.has_edge(pu, pv)
    assert hw_graph.has_edge(pv, pv2)


def test_place_vf2_returns_none_when_no_exact_match_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)  # 0-1-2, max degree 2
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)
    interaction.add_edge(1, 2, weight=1)  # a triangle -- no triangle exists in a line graph

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality)

    assert result is None


def test_place_vf2_respects_zero_time_budget_by_returning_none():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place_vf2(hw_graph, interaction, quality, time_budget_seconds=0.0)

    assert result is None


def test_place_dispatcher_falls_back_to_greedy_when_vf2_has_no_match():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(3)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    interaction.add_edge(0, 2, weight=1)
    interaction.add_edge(1, 2, weight=1)  # triangle, impossible on a line

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place(hw_graph, interaction, quality)

    assert result.strategy_used == "greedy"
    assert len(result.assignment) == 3


def test_place_dispatcher_prefers_vf2_when_exact_match_exists():
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw_graph = _line_hardware_graph(5)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)

    quality = HardwareQualityModel(hw_graph, objective="balanced")
    result = place(hw_graph, interaction, quality)

    assert result.strategy_used == "vf2"
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `pytest tests/test_ibm_placement.py -v`
Expected: 5 new `FAIL`s (`ImportError: cannot import name 'place_vf2'`), 3 prior `PASS`

- [ ] **Step 3: Add `place_vf2` and `place` to `_placement.py`**

Append to the file from Task 8 (keep everything already there):

```python
import time
from statistics import mean
from typing import Optional

from networkx.algorithms.isomorphism import GraphMatcher

_VF2_MAX_INTERACTION_NODES = 20  # cheap pre-filter only; the time budget below is the real bound
_VF2_DEFAULT_TIME_BUDGET_SECONDS = 2.0


def place_vf2(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    quality: HardwareQualityModel,
    *,
    time_budget_seconds: float = _VF2_DEFAULT_TIME_BUDGET_SECONDS,
) -> Optional[PlacementResult]:
    """Exact subgraph monomorphism search: find a physical region where every
    logical interaction maps to an existing physical coupler (zero predicted
    SWAPs). Bounded by wall-clock time, not a candidate-count guess -- fixes
    cahgr/retrieval.py's _VF2_NODE_CAP/_VF2_CANDIDATE_CAP, which capped
    already-yielded candidates but had no bound on time spent before the first
    yield. Returns None (never raises) if no match is found or the budget runs
    out; callers should fall back to place_greedy."""
    if interaction_graph.number_of_nodes() == 0 or interaction_graph.number_of_nodes() > _VF2_MAX_INTERACTION_NODES:
        return None
    if interaction_graph.number_of_edges() == 0:
        return None  # nothing to align structurally

    matcher = GraphMatcher(hw_graph, interaction_graph)
    best_assignment: dict[int, int] | None = None
    best_score = float("-inf")
    deadline = time.monotonic() + time_budget_seconds

    for mapping in matcher.subgraph_monomorphisms_iter():
        if time.monotonic() >= deadline:
            break
        assignment = {logical_q: phys_q for phys_q, logical_q in mapping.items()}
        score = _score_assignment(hw_graph, interaction_graph, quality, assignment)
        if score > best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        return None
    return PlacementResult(assignment=best_assignment, strategy_used="vf2")


def place(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    quality: HardwareQualityModel,
    *,
    connectivity_weight: float = 0.7,
    calibration_weight: float = 0.3,
) -> PlacementResult:
    """Try an exact VF2 match first (zero-SWAP result when possible), falling
    back to greedy region-growing, which always succeeds."""
    result = place_vf2(hw_graph, interaction_graph, quality)
    if result is not None:
        return result
    return place_greedy(
        hw_graph,
        interaction_graph,
        quality,
        connectivity_weight=connectivity_weight,
        calibration_weight=calibration_weight,
    )


def _score_assignment(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    quality: HardwareQualityModel,
    assignment: dict[int, int],
) -> float:
    node_component = mean(quality.node_score(p) for p in assignment.values()) if assignment else 0.0
    edge_scores = []
    for u, v, d in interaction_graph.edges(data=True):
        pu, pv = assignment.get(u), assignment.get(v)
        if pu is None or pv is None:
            continue
        if hw_graph.has_edge(pu, pv):
            edge_scores.append(d["weight"] * quality.edge_score(pu, pv))
    edge_component = sum(edge_scores) / max(1, interaction_graph.number_of_edges())
    return 0.3 * node_component + 0.7 * edge_component
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ibm_placement.py -v`
Expected: `PASS` (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_placement.py tests/test_ibm_placement.py
git commit -m "feat(ibm): add time-budgeted VF2 placement, fixing the intern's uncapped-time defect"
```

---

### Task 10: `_coupling_diagram.py` — coupling-map SVG

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/_coupling_diagram.py`
- Test: `tests/test_ibm_coupling_diagram.py`

**Interfaces:**
- Consumes: a `networkx.Graph` (the hardware graph from Task 4) and a `dict[int, int]`
  (logical→physical assignment from Task 8/9, used only for its `.values()`).
- Produces: `render_coupling_diagram(hw_graph: nx.Graph, highlighted_qubits:
  set[int]) -> str` (an SVG string, empty string `""` on any rendering failure — never
  raises). Consumed by Task 11 (`map_circuit.py`).

**Pattern source:** matplotlib→SVG-string-with-best-effort-fallback, matching
`threshold_sweep._render_plot`'s existing pattern in this repo (`except Exception: svg = ""`,
described there as "plot is best-effort"). Layout uses `networkx.kamada_kawai_layout` instead
of the intern's per-chip coordinate table (see Task 4's note).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_coupling_diagram.py
from __future__ import annotations

import networkx as nx


def test_render_coupling_diagram_returns_svg_string():
    from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    svg = render_coupling_diagram(g, highlighted_qubits={0, 1})

    assert svg.startswith("<?xml") or "<svg" in svg


def test_render_coupling_diagram_handles_empty_graph_without_raising():
    from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram

    svg = render_coupling_diagram(nx.Graph(), highlighted_qubits=set())

    assert isinstance(svg, str)  # "" on failure, or a valid (if trivial) SVG -- never raises
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_coupling_diagram.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `_coupling_diagram.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/_coupling_diagram.py
"""Coupling-map -> SVG, mapped qubits highlighted.

Follows this repo's existing matplotlib -> SVG-string, best-effort-fallback
pattern (see threshold_sweep._render_plot: "plot is best-effort"). Uses
networkx's own kamada_kawai_layout instead of porting the intern's per-chip
coordinate table (hardware_layout.qubit_coordinates) -- unnecessary for a
"which qubits were picked" diagram.
"""

from __future__ import annotations

import io

import networkx as nx


def render_coupling_diagram(hw_graph: nx.Graph, highlighted_qubits: set[int]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if hw_graph.number_of_nodes() == 0:
            return ""

        pos = nx.kamada_kawai_layout(hw_graph)
        fig, ax = plt.subplots(figsize=(6, 6))
        node_colors = ["#d62728" if n in highlighted_qubits else "#cccccc" for n in hw_graph.nodes]
        nx.draw(
            hw_graph,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            with_labels=True,
            node_size=300,
            font_size=8,
        )
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_coupling_diagram.py -v`
Expected: `PASS` (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec_mcp_tools/tools/ibm_hardware/_coupling_diagram.py tests/test_ibm_coupling_diagram.py
git commit -m "feat(ibm): render coupling-map SVG with mapped qubits highlighted"
```

---

### Task 11: `ibm_map_circuit` tool

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py`
- Modify: `src/chatqec_mcp_tools/server.py`
- Test: `tests/test_ibm_map_circuit.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `get_ibm_service`/`IBMCredentialsError` (Task 1), `get_or_fetch` (Task 2, for the
  backend-list step of auto-selection), `build_hardware_graph` (Task 4), `HardwareQualityModel`
  (Task 6), `parse_circuit_input`/`build_interaction_graph` (Task 7), `place` (Task 9),
  `render_coupling_diagram` (Task 10).
- Produces: `IbmMapCircuitInput` (fields: `circuit_qasm: str | None = None,
  circuit_preset: str | None = None, num_qubits: int | None = None, qaoa_layers: int | None =
  None, backend: str | None = None, objective: str = "maximum_fidelity"`),
  `IbmMapCircuitOutput` (fields: `ok, backend, physical_qubits: list[int],
  logical_to_physical: dict[str, int], strategy_used, transpiled_two_qubit_gate_count: int,
  estimated_fidelity: float, coupling_map_svg: str, fetched_at, age_seconds, error_message`),
  `ibm_map_circuit(inp) -> IbmMapCircuitOutput`. Also produces a reusable helper
  `_select_backend_and_map(inp) -> MappingOutcome` consumed directly by Tasks 12 and 13 so
  preview/submit don't duplicate this logic.

*Note on `logical_to_physical` typing:* pydantic/JSON requires string dict keys, so this
field is `dict[str, int]` even though the underlying assignment is `dict[int, int]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_map_circuit.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def test_ibm_map_circuit_success_with_explicit_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
    ):
        out = ibm_map_circuit(
            IbmMapCircuitInput(circuit_preset="bell", backend="ibm_fake", objective="maximum_fidelity")
        )

    assert out.ok is True
    assert out.backend == "ibm_fake"
    assert len(out.physical_qubits) == 2
    assert out.strategy_used in ("vf2", "greedy")
    assert 0.0 <= out.estimated_fidelity <= 1.0
    assert out.coupling_map_svg != ""


def test_ibm_map_circuit_auto_selects_least_busy_operational_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    busy = SimpleNamespace(
        name="ibm_busy", num_qubits=5, status=lambda: SimpleNamespace(operational=True, pending_jobs=9)
    )
    quiet = SimpleNamespace(
        name="ibm_quiet", num_qubits=5, status=lambda: SimpleNamespace(operational=True, pending_jobs=0)
    )
    down = SimpleNamespace(
        name="ibm_down", num_qubits=5, status=lambda: SimpleNamespace(operational=False, pending_jobs=0)
    )
    fake_service = SimpleNamespace(backends=lambda: [busy, down, quiet])

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
    ):
        out = ibm_map_circuit(IbmMapCircuitInput(circuit_preset="bell"))  # no backend given

    assert out.ok is True
    assert out.backend == "ibm_quiet"


def test_ibm_map_circuit_rejects_both_qasm_and_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    out = ibm_map_circuit(
        IbmMapCircuitInput(circuit_qasm="OPENQASM 2.0;", circuit_preset="bell", backend="ibm_fake")
    )

    assert out.ok is False
    assert "exactly one" in out.error_message


def test_ibm_map_circuit_rejects_circuit_too_big_for_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph(n=2)),
    ):
        out = ibm_map_circuit(IbmMapCircuitInput(circuit_preset="ghz", num_qubits=5, backend="ibm_fake"))

    assert out.ok is False
    assert "qubits" in out.error_message.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_map_circuit.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `map_circuit.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py
"""ibm_map_circuit tool -- map a circuit onto a chosen or auto-selected IBM
backend using the ported quality-scored placement engine."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
    build_interaction_graph,
    parse_circuit_input,
)
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram
from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph
from chatqec_mcp_tools.tools.ibm_hardware._placement import PlacementResult, place
from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

_VALID_OBJECTIVES = ("maximum_fidelity", "minimum_execution_time")


class IbmMapCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = None
    qaoa_layers: int | None = None
    backend: str | None = None
    objective: str = "maximum_fidelity"


class IbmMapCircuitOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    physical_qubits: list[int] = []
    logical_to_physical: dict[str, int] = {}
    strategy_used: str = ""
    transpiled_two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    coupling_map_svg: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    error_message: str = ""


@dataclass
class MappingOutcome:
    """Shared result shape reused by ibm_preview_execution / ibm_submit_execution
    so they don't duplicate this tool's mapping logic."""

    ok: bool
    backend: str = ""
    placement: PlacementResult | None = None
    two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    svg: str = ""
    error_message: str = ""


def _auto_select_backend(service, *, min_qubits: int):
    """Pick the operational backend with the fewest pending jobs among those
    with enough qubits; ties broken by name for determinism. Returns None if
    no backend qualifies."""
    candidates = []
    for backend in service.backends():
        if backend.num_qubits < min_qubits:
            continue
        status = backend.status()
        if not status.operational:
            continue
        candidates.append((status.pending_jobs, backend.name, backend))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def compute_mapping(
    *,
    circuit_qasm: str | None,
    circuit_preset: str | None,
    num_qubits: int | None,
    qaoa_layers: int | None,
    backend: str | None,
    objective: str,
    include_svg: bool = True,
) -> MappingOutcome:
    if objective not in _VALID_OBJECTIVES:
        return MappingOutcome(ok=False, error_message=f"objective must be one of {_VALID_OBJECTIVES}")

    try:
        spec = parse_circuit_input(
            circuit_qasm=circuit_qasm,
            circuit_preset=circuit_preset,
            num_qubits=num_qubits,
            qaoa_layers=qaoa_layers,
        )
    except ValueError as exc:
        return MappingOutcome(ok=False, error_message=str(exc))

    try:
        service = get_ibm_service()
        if backend:
            sdk_backend = service.backend(backend)
        else:
            sdk_backend = _auto_select_backend(service, min_qubits=spec.num_logical_qubits)
            if sdk_backend is None:
                return MappingOutcome(
                    ok=False,
                    error_message=(
                        f"No operational backend with at least {spec.num_logical_qubits} "
                        "qubits is visible to this account."
                    ),
                )
        hw_graph = build_hardware_graph(sdk_backend)
    except IBMCredentialsError as exc:
        return MappingOutcome(ok=False, error_message=str(exc))
    except Exception as exc:
        return MappingOutcome(ok=False, error_message=f"Failed to load backend {backend}: {exc}")

    backend = sdk_backend.name

    if spec.num_logical_qubits > hw_graph.num_qubits:
        return MappingOutcome(
            ok=False,
            error_message=(
                f"Circuit needs {spec.num_logical_qubits} qubits but {backend} "
                f"has only {hw_graph.num_qubits}."
            ),
        )

    interaction = build_interaction_graph(spec.circuit)
    quality = HardwareQualityModel(hw_graph.graph, objective=objective)
    placement = place(hw_graph.graph, interaction.graph, quality)

    two_q_count = sum(1 for instr in spec.circuit.data if len(instr.qubits) == 2)
    fidelities = [
        quality.edge_score(placement.assignment[u], placement.assignment[v])
        for u, v in interaction.graph.edges
        if u in placement.assignment and v in placement.assignment
    ]
    estimated_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0

    svg = ""
    if include_svg:
        svg = render_coupling_diagram(hw_graph.graph, set(placement.assignment.values()))

    return MappingOutcome(
        ok=True,
        backend=backend,
        placement=placement,
        two_qubit_gate_count=two_q_count,
        estimated_fidelity=estimated_fidelity,
        svg=svg,
    )


def ibm_map_circuit(inp: IbmMapCircuitInput) -> IbmMapCircuitOutput:
    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective=inp.objective,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmMapCircuitOutput(ok=False, error_message=outcome.error_message)

    return IbmMapCircuitOutput(
        ok=True,
        backend=outcome.backend,
        physical_qubits=sorted(outcome.placement.assignment.values()),
        logical_to_physical={str(k): v for k, v in outcome.placement.assignment.items()},
        strategy_used=outcome.placement.strategy_used,
        transpiled_two_qubit_gate_count=outcome.two_qubit_gate_count,
        estimated_fidelity=outcome.estimated_fidelity,
        coupling_map_svg=outcome.svg,
        fetched_at=dt.datetime.now(tz=dt.UTC).isoformat(),
        age_seconds=0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_map_circuit.py -v`
Expected: `PASS` (4 tests)

- [ ] **Step 5: Register in `server.py`** (same pattern as before; description: *"Map a
circuit (QASM or a bell/ghz/qft/qaoa preset) onto an IBM backend: physical qubit layout,
strategy used, estimated fidelity, and a coupling-map diagram. Free, local compute plus one
calibration fetch."*), add to `test_server.py`.

- [ ] **Step 6: Run full suite, then commit**

```bash
pytest tests/test_ibm_map_circuit.py tests/test_server.py -v
git add src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py \
        src/chatqec_mcp_tools/server.py \
        tests/test_ibm_map_circuit.py tests/test_server.py
git commit -m "feat(ibm): add ibm_map_circuit tool"
```

---

### Task 12: `ibm_preview_execution` tool

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/preview_execution.py`
- Modify: `src/chatqec_mcp_tools/server.py`
- Test: `tests/test_ibm_preview_execution.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `compute_mapping`/`MappingOutcome` from Task 11's `map_circuit.py`, `get_ibm_service`
  (Task 1).
- Produces: `IbmPreviewExecutionInput` (same circuit fields as `IbmMapCircuitInput` plus
  `shots: int = 1000`), `IbmPreviewExecutionOutput` (`ok, backend, physical_qubits,
  strategy_used, estimated_fidelity, shots, estimated_queue_position: int | None,
  backend_operational: bool, error_message`), `ibm_preview_execution(inp) -> Output`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_preview_execution.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def test_ibm_preview_execution_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
        IbmPreviewExecutionInput,
        ibm_preview_execution,
    )

    fake_service = SimpleNamespace(
        backend=lambda name: SimpleNamespace(
            name=name, status=lambda: SimpleNamespace(operational=True, pending_jobs=2)
        )
    )

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.preview_execution.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
    ):
        out = ibm_preview_execution(
            IbmPreviewExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=500)
        )

    assert out.ok is True
    assert out.shots == 500
    assert out.estimated_queue_position == 2
    assert out.backend_operational is True
    assert len(out.physical_qubits) == 2


def test_ibm_preview_execution_propagates_mapping_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
        IbmPreviewExecutionInput,
        ibm_preview_execution,
    )

    out = ibm_preview_execution(IbmPreviewExecutionInput(backend="ibm_fake"))  # no circuit given

    assert out.ok is False
    assert "exactly one" in out.error_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_preview_execution.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `preview_execution.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/preview_execution.py
"""ibm_preview_execution tool -- everything ibm_submit_execution would do,
minus submission. Free; call this before ibm_submit_execution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import compute_mapping


class IbmPreviewExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = None
    qaoa_layers: int | None = None
    backend: str
    objective: str = "maximum_fidelity"
    shots: int = 1000


class IbmPreviewExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    physical_qubits: list[int] = []
    strategy_used: str = ""
    estimated_fidelity: float = 0.0
    shots: int = 0
    estimated_queue_position: int | None = None
    backend_operational: bool = False
    error_message: str = ""


def ibm_preview_execution(inp: IbmPreviewExecutionInput) -> IbmPreviewExecutionOutput:
    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective=inp.objective,
        include_svg=False,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmPreviewExecutionOutput(ok=False, error_message=outcome.error_message)

    try:
        service = get_ibm_service()
        status = service.backend(inp.backend).status()
        operational, pending_jobs = status.operational, status.pending_jobs
    except IBMCredentialsError as exc:
        return IbmPreviewExecutionOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IbmPreviewExecutionOutput(ok=False, error_message=f"Failed to check backend status: {exc}")

    return IbmPreviewExecutionOutput(
        ok=True,
        backend=outcome.backend,
        physical_qubits=sorted(outcome.placement.assignment.values()),
        strategy_used=outcome.placement.strategy_used,
        estimated_fidelity=outcome.estimated_fidelity,
        shots=inp.shots,
        estimated_queue_position=pending_jobs,
        backend_operational=operational,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_preview_execution.py -v`
Expected: `PASS` (2 tests)

- [ ] **Step 5: Register in `server.py`** (description: *"Preview a real-hardware submission
— mapped qubits, backend status, shots — without submitting or spending credits. Call this
before ibm_submit_execution."*), add to `test_server.py`.

- [ ] **Step 6: Run full suite, then commit**

```bash
pytest tests/test_ibm_preview_execution.py tests/test_server.py -v
git add src/chatqec_mcp_tools/tools/ibm_hardware/preview_execution.py \
        src/chatqec_mcp_tools/server.py \
        tests/test_ibm_preview_execution.py tests/test_server.py
git commit -m "feat(ibm): add ibm_preview_execution tool"
```

---

### Task 13: `ibm_submit_execution` tool

**Files:**
- Create: `src/chatqec_mcp_tools/tools/ibm_hardware/submit_execution.py`
- Modify: `src/chatqec_mcp_tools/server.py`
- Test: `tests/test_ibm_submit_execution.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `compute_mapping` (Task 11), `get_ibm_service`/`IBMCredentialsError` (Task 1).
- Produces: `IbmSubmitExecutionInput` (`circuit_qasm | circuit_preset` fields, `backend: str`
  [required, no default], `shots: int`), `IbmSubmitExecutionOutput` (`ok, job_id, backend,
  shots, status, error_message`), `ibm_submit_execution(inp) -> Output`.

**Caps enforced here, per the spec:** `num_qubits <= 10`, `shots <= 4000`, `backend` required
— checked before any SDK call, all violations return `ok=False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ibm_submit_execution.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def test_ibm_submit_execution_rejects_missing_backend():
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(circuit_preset="bell", shots=100)  # backend omitted


def test_ibm_submit_execution_rejects_too_many_shots(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    out = ibm_submit_execution(
        IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=5000)
    )

    assert out.ok is False
    assert "shots" in out.error_message.lower()


def test_ibm_submit_execution_rejects_too_many_qubits(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    out = ibm_submit_execution(
        IbmSubmitExecutionInput(circuit_preset="ghz", num_qubits=11, backend="ibm_fake", shots=100)
    )

    assert out.ok is False
    assert "qubit" in out.error_message.lower()


def test_ibm_submit_execution_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    fake_job = SimpleNamespace(job_id=lambda: "job-123", status=lambda: "QUEUED")
    fake_sampler = SimpleNamespace(run=lambda *a, **kw: fake_job)
    fake_service = SimpleNamespace(backend=lambda name: SimpleNamespace(name=name))

    with (
        patch("chatqec_mcp_tools.tools.ibm_hardware.submit_execution.get_ibm_service", return_value=fake_service),
        patch("chatqec_mcp_tools.tools.ibm_hardware.map_circuit.build_hardware_graph", return_value=_fake_hw_graph()),
        patch("chatqec_mcp_tools.tools.ibm_hardware.submit_execution._make_sampler", return_value=fake_sampler),
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    assert out.job_id == "job-123"
    assert out.status == "QUEUED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ibm_submit_execution.py -v`
Expected: `FAIL` with `ModuleNotFoundError`

- [ ] **Step 3: Implement `submit_execution.py`**

```python
# src/chatqec_mcp_tools/tools/ibm_hardware/submit_execution.py
"""ibm_submit_execution tool -- SPENDS REAL CREDITS. Caps enforced before any
SDK call: num_qubits <= 10, shots <= 4000, backend required (no auto-pick).
Call ibm_preview_execution first."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import compute_mapping

_MAX_QUBITS = 10
_MAX_SHOTS = 4000


class IbmSubmitExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = None
    qaoa_layers: int | None = None
    backend: str = Field(..., min_length=1)
    shots: int


class IbmSubmitExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    job_id: str = ""
    backend: str = ""
    shots: int = 0
    status: str = ""
    error_message: str = ""


def _make_sampler(service: Any, backend: Any) -> Any:
    from qiskit_ibm_runtime import SamplerV2  # noqa: PLC0415

    return SamplerV2(mode=backend)


def ibm_submit_execution(inp: IbmSubmitExecutionInput) -> IbmSubmitExecutionOutput:
    if inp.shots > _MAX_SHOTS:
        return IbmSubmitExecutionOutput(
            ok=False, error_message=f"shots ({inp.shots}) exceeds the cap of {_MAX_SHOTS}."
        )
    if inp.num_qubits is not None and inp.num_qubits > _MAX_QUBITS:
        return IbmSubmitExecutionOutput(
            ok=False, error_message=f"num_qubits ({inp.num_qubits}) exceeds the cap of {_MAX_QUBITS}."
        )

    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective="maximum_fidelity",
        include_svg=False,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmSubmitExecutionOutput(ok=False, error_message=outcome.error_message)

    if len(outcome.placement.assignment) > _MAX_QUBITS:
        return IbmSubmitExecutionOutput(
            ok=False,
            error_message=(
                f"Circuit maps to {len(outcome.placement.assignment)} physical qubits, "
                f"which exceeds the cap of {_MAX_QUBITS}."
            ),
        )

    try:
        service = get_ibm_service()
        sdk_backend = service.backend(inp.backend)
        sampler = _make_sampler(service, sdk_backend)
        job = sampler.run([], shots=inp.shots)
    except IBMCredentialsError as exc:
        return IbmSubmitExecutionOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IbmSubmitExecutionOutput(ok=False, error_message=f"Submission failed: {exc}")

    return IbmSubmitExecutionOutput(
        ok=True,
        job_id=job.job_id(),
        backend=inp.backend,
        shots=inp.shots,
        status=str(job.status()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ibm_submit_execution.py -v`
Expected: `PASS` (4 tests)

- [ ] **Step 5: Register in `server.py`** (description: *"SUBMIT a real job to IBM Quantum
hardware — spends account credits/quota. Requires an explicit backend name (no auto-pick),
caps at 10 qubits / 4000 shots. Call ibm_preview_execution first to see the plan before
spending anything."*), add to `test_server.py`.

- [ ] **Step 6: Run full suite, then commit**

```bash
pytest tests/test_ibm_submit_execution.py tests/test_server.py -v
git add src/chatqec_mcp_tools/tools/ibm_hardware/submit_execution.py \
        src/chatqec_mcp_tools/server.py \
        tests/test_ibm_submit_execution.py tests/test_server.py
git commit -m "feat(ibm): add ibm_submit_execution tool with qubit/shot/backend caps"
```

---

### Task 14: README + full-suite verification

**Files:**
- Modify: `README.md` (in `chatqec-mcp-tools`)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Read the existing README's tool-list section** to match its exact style
(look for how `stim_simulate`, `code_params`, etc. are documented, including the `[lightstim]`
/`[tsim]` optional-extra install instructions).

- [ ] **Step 2: Add a new section documenting the 5 IBM tools**, following the same style,
covering: the `ibm` extra install command (`uv tool install --force '~/chatqec-mcp-tools[ibm]'`),
the two env vars (`QISKIT_IBM_TOKEN` required, `QISKIT_IBM_INSTANCE` optional), a one-line
description of each of the 5 tools, and an explicit warning that `ibm_submit_execution`
spends real account credits and should only be reachable after `ibm_preview_execution`.

- [ ] **Step 3: Run the complete test suite for the whole repo**

Run: `pytest -v`
Expected: all tests `PASS`, including every test from Tasks 1–13 plus all pre-existing tests
unaffected.

- [ ] **Step 4: Run lint/type checks** (matching this repo's pre-commit hooks)

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: clean; fix any findings before proceeding.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the 5 new IBM hardware-aware MCP tools"
```

---

## Self-Review Notes (for whoever executes this plan)

- **Spec coverage:** All 5 tools (Tasks 3, 5, 11, 12, 13), caching (Task 2), credentials
  (Task 1), the VF2 fix (Task 9), the coupling diagram (Task 10), and the `ibm` extra
  (Task 1) are covered. The spec's "Known limitations" section (preview-first is a soft
  guarantee, TTL is a starting guess) are accepted as-is — no task tries to build a
  session/state layer to hard-enforce preview-before-submit, matching the spec's explicit
  decision.
- **Not covered by this plan, and intentionally so per the spec's "Out of scope" section:**
  Steiner-tree placement, a UI job-history view, multi-backend comparison in one call, IQM
  support.
- **`chatQEC` itself needs no changes** — confirmed in the spec: `conf/config.yaml` already
  points at `chatqec-mcp-tools` as `server_command`, so upgrading the installed package via
  `uv tool install --force` is sufficient for the new tools to reach the tool loop.
