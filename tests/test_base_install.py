"""Base-install regression guard (no `ibm` extra installed).

`server.py` imports all 13 tool modules unconditionally to register them, so
ANY module-scope import of an extra-only package in `tools/ibm_hardware/`
breaks the whole MCP server -- not just the 5 IBM tools, but the 8
pre-existing ones too. That is exactly what a module-scope `import rustworkx`
in `_placement.py` did.

Two complementary layers here:

1. A subprocess test that blocks rustworkx/qiskit/qiskit_ibm_runtime at
   import time for the WHOLE module tree and imports `chatqec_mcp_tools.server`
   from scratch. A subprocess is used deliberately: the pytest process has
   already imported `chatqec_mcp_tools.server` (and qiskit, via other test
   modules), so an in-process `importlib.reload` could not give a genuinely
   fresh import of the entire tree, and tearing `chatqec_mcp_tools` out of
   sys.modules mid-session would leave duplicate module/class objects that
   other tests' `patch("...map_circuit.get_ibm_service")` targets would miss.
   The subprocess is the only way to test the real condition honestly.

2. In-process tests that block the same modules only at call time (via
   `builtins.__import__`, the convention `test_ibm_client.py` already uses),
   asserting each of the 5 IBM tools returns ok=False with a helpful message
   instead of raising ModuleNotFoundError. Call-time blocking is sufficient
   there precisely because every qiskit import in this package is lazy.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_BLOCKED = ("rustworkx", "qiskit", "qiskit_ibm_runtime", "qiskit_paulice", "qiskit_aer", "deltakit", "guppylang", "selene_sim", "ft_primitive_bench", "relay_bp")

_EXPECTED_TOOLS = {
    # the 9 non-IBM tools -- these are the collateral damage a module-scope
    # extra-only import causes, and the reason this is a regression not a gap
    "code_params",
    "glcb_visualize_url",
    "pymatching_decode",
    "qec_circuit_build",
    "stim_diagram",
    "stim_simulate",
    "threshold_sweep",
    "tsim_simulate",
    "decoder_compare",
    "ft_primitive_simulate",
    # the 5 IBM tools
    "ibm_backend_calibration",
    "ibm_list_backends",
    "ibm_map_circuit",
    "ibm_preview_execution",
    "ibm_submit_execution",
    "ibm_add_pauli_checks",
    "ibm_ft_compile",
    "deltakit_memory",
    "guppy_run",
}

# Runs in a fresh interpreter with no `ibm` extra visible. A meta_path finder
# raising ImportError is a truer simulation than a sys.modules stub: it fails
# the same way a genuinely absent package does, for every importer in the tree.
_CHILD = textwrap.dedent(
    """
    import json, sys

    BLOCKED = {blocked!r}

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root in BLOCKED:
                raise ImportError("No module named %r (simulated base install)" % root)
            return None

    sys.meta_path.insert(0, _Blocker())
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]

    from chatqec_mcp_tools.server import build_server

    # The specific module that regressed, imported without the extra present.
    import chatqec_mcp_tools.tools.ibm_hardware._placement  # noqa: F401

    print(json.dumps({{
        "tools": sorted(t.name for t in build_server().list_tools()),
        # Proof the lazy import really stayed lazy: importing the whole server
        # must not have pulled any blocked package into sys.modules.
        "leaked": sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED),
    }}))
    """
).format(blocked=set(_BLOCKED))


def test_server_imports_and_registers_all_tools_without_the_ibm_extra(tmp_path):
    """CORE REGRESSION TEST for the base install.

    Before the fix this failed at import time with
    `ModuleNotFoundError: No module named 'rustworkx'` raised from
    `_placement.py`'s module scope, via server.py -> map_circuit -> _placement,
    taking down all 13 tools on the README's own primary install command.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env={"XDG_CACHE_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        check=False,
    )

    assert proc.returncode == 0, (
        "importing chatqec_mcp_tools.server failed without the 'ibm' extra:\n" + proc.stderr
    )
    result = json.loads(proc.stdout)
    assert set(result["tools"]) == _EXPECTED_TOOLS
    assert result["leaked"] == [], (
        "an extra-only package was imported while building the server: " f"{result['leaked']}"
    )


# --------------------------------------------------------------------------
# Per-tool graceful degradation when the SDK is missing.
# --------------------------------------------------------------------------


def _block_imports(monkeypatch, *names):
    """Make `names` (and their submodules) unimportable for the duration of a
    test, passing every other import through. Same shape as
    test_ibm_client.py's `_raising_import`, generalized to several names."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in names:
            raise ImportError(f"No module named '{root}'")
        return real_import(name, *args, **kwargs)

    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)


def _tool_cases():
    from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
        IbmBackendCalibrationInput,
        ibm_backend_calibration,
    )
    from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
        IbmListBackendsInput,
        ibm_list_backends,
    )
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )
    from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
        IbmPreviewExecutionInput,
        ibm_preview_execution,
    )
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    return {
        "ibm_list_backends": (ibm_list_backends, IbmListBackendsInput()),
        "ibm_backend_calibration": (
            ibm_backend_calibration,
            IbmBackendCalibrationInput(backend="ibm_fake"),
        ),
        "ibm_map_circuit": (ibm_map_circuit, IbmMapCircuitInput(circuit_preset="bell")),
        "ibm_preview_execution": (
            ibm_preview_execution,
            IbmPreviewExecutionInput(circuit_preset="bell", backend="ibm_fake"),
        ),
        "ibm_submit_execution": (
            ibm_submit_execution,
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=10),
        ),
    }


@pytest.mark.parametrize(
    "tool_name",
    [
        "ibm_list_backends",
        "ibm_backend_calibration",
        "ibm_map_circuit",
        "ibm_preview_execution",
        "ibm_submit_execution",
    ],
)
def test_ibm_tool_degrades_gracefully_without_the_sdk(tool_name, monkeypatch, tmp_path):
    """Every IBM tool must honour the feature-wide contract -- ok=False plus a
    non-empty error_message, nothing raises -- when the 'ibm' extra is absent.

    Before the fix, ibm_map_circuit / ibm_preview_execution /
    ibm_submit_execution raised a raw ModuleNotFoundError out of
    `_circuits.parse_circuit_input`'s lazy `from qiskit import QuantumCircuit`,
    because compute_mapping caught only (ValueError, OverflowError,
    ArithmeticError) around that call.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    fn, inp = _tool_cases()[tool_name]

    # Inputs are built BEFORE blocking, so a pydantic-side import can't confuse
    # what this test is measuring: only the tool call runs SDK-less.
    _block_imports(monkeypatch, *_BLOCKED)
    out = fn(inp)

    assert out.ok is False
    assert out.error_message, f"{tool_name} returned ok=False with no error_message"
    # The message must be actionable, i.e. name the extra to install -- the
    # single canonical hint from _client.IBM_EXTRA_INSTALL_HINT.
    assert "ibm" in out.error_message.lower()


def test_error_message_uses_the_canonical_install_hint(monkeypatch, tmp_path):
    """compute_mapping's ImportError message must reuse _client.py's existing
    install-hint wording rather than inventing its own, so the two SDK-missing
    paths (qiskit vs qiskit-ibm-runtime) read consistently."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware._client import IBM_EXTRA_INSTALL_HINT
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    inp = IbmMapCircuitInput(circuit_preset="bell")
    _block_imports(monkeypatch, *_BLOCKED)
    out = ibm_map_circuit(inp)

    assert out.ok is False
    assert IBM_EXTRA_INSTALL_HINT in out.error_message


def test_place_vf2_imports_rustworkx_lazily(monkeypatch):
    """Targeted proof that the rustworkx import moved into the function body:
    with rustworkx blocked, `_placement` is fully usable except for the one
    function that needs it, and `place()` still succeeds by falling back to
    place_greedy -- which is the whole point of keeping rustworkx extra-only."""
    import networkx as nx

    from chatqec_mcp_tools.tools.ibm_hardware._placement import place, place_vf2
    from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

    hw = nx.Graph()
    for q in range(4):
        hw.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(3):
        hw.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    interaction = nx.Graph()
    interaction.add_edge(0, 1, weight=1)
    quality = HardwareQualityModel(hw, objective="maximum_fidelity")

    _block_imports(monkeypatch, "rustworkx")

    # place_vf2 is the only consumer, so it is the only thing that notices.
    with pytest.raises(ImportError):
        place_vf2(hw, interaction, quality)

    # The module-level contract that matters -- importing and using the rest of
    # _placement without rustworkx present -- holds.
    from chatqec_mcp_tools.tools.ibm_hardware._placement import place_greedy

    result = place_greedy(hw, interaction, quality)
    assert set(result.assignment) == {0, 1}
    assert result.strategy_used == "greedy"
    assert place is not None


def test_compute_mapping_survives_rustworkx_missing_but_qiskit_present(monkeypatch, tmp_path):
    """Moving the rustworkx import into place_vf2 turns an import-time crash
    into a call-time one for the (unsupported, since the 'ibm' extra installs
    both) partial install where qiskit is present but rustworkx is not. Confirm
    that still honours the ok=False contract rather than raising: place()'s
    ImportError lands in compute_mapping's existing `except Exception` around
    the placement block."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
        IbmMapCircuitInput,
        ibm_map_circuit,
    )

    hw = __import__("networkx").Graph()
    for q in range(4):
        hw.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(3):
        hw.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)

    inp = IbmMapCircuitInput(circuit_preset="bell", backend="ibm_fake")
    module = "chatqec_mcp_tools.tools.ibm_hardware.map_circuit"
    with (
        patch(
            f"{module}.get_ibm_service",
            return_value=SimpleNamespace(backend=lambda name: SimpleNamespace(name=name)),
        ),
        patch(
            f"{module}.build_hardware_graph",
            return_value=SimpleNamespace(graph=hw, backend_name="ibm_fake"),
        ),
    ):
        _block_imports(monkeypatch, "rustworkx")
        out = ibm_map_circuit(inp)

    assert out.ok is False
    assert out.error_message
