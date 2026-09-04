from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from pydantic import ValidationError

_MAP_MODULE = "chatqec_mcp_tools.tools.ibm_hardware.map_circuit"
_SUBMIT_MODULE = "chatqec_mcp_tools.tools.ibm_hardware.submit_execution"


def _fake_hw_graph(n=5):
    g = nx.Graph()
    for q in range(n):
        g.add_node(q, T1=50.0, T2=45.0, readout_fidelity=0.99, is_operational=True)
    for q in range(n - 1):
        g.add_edge(q, q + 1, gate_fidelity=0.99, gate_time=68.0)
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def _fake_service():
    """A fake service plus the list of backend objects it has handed out.

    Every .backend() call returns a DISTINCT object (recorded in that list), so
    an `is resolved[0]` assertion genuinely pins the sampler/transpile to the one
    backend compute_mapping resolved -- a regression that re-fetched the backend
    would get resolved[1] and fail. A service that returned one shared object
    would make such an assertion vacuous."""
    resolved: list[SimpleNamespace] = []

    def _backend(name):
        obj = SimpleNamespace(name=name)
        resolved.append(obj)
        return obj

    return SimpleNamespace(backend=_backend), resolved


def _skewed_hw_graph(n=5):
    """Like _fake_hw_graph but with the far end of the line clearly the best
    place to sit, so the resolved placement is NOT the identity [0, 1, ...].
    Without that skew an `initial_layout == placement` assertion is vacuous: a
    regression that ignored the placement entirely and passed list(range(n))
    would still match."""
    g = nx.Graph()
    for q in range(n):
        good = q >= n - 2
        g.add_node(
            q,
            T1=200.0 if good else 20.0,
            T2=180.0 if good else 18.0,
            readout_fidelity=0.999 if good else 0.90,
            is_operational=True,
        )
    for q in range(n - 1):
        good = q == n - 2
        g.add_edge(
            q, q + 1, gate_fidelity=0.999 if good else 0.90, gate_time=40.0 if good else 400.0
        )
    return SimpleNamespace(graph=g, backend_name="ibm_fake")


def _fake_job(job_id="job-123", status="QUEUED"):
    return SimpleNamespace(job_id=lambda: job_id, status=lambda: status)


@contextlib.contextmanager
def _patched_sdk(service, *, sampler=None, isa_circuit=None, hw_graph=None):
    """Patch every SDK boundary this tool can reach.

    compute_mapping lives in map_circuit.py and calls get_ibm_service() using THAT
    module's imported name -- patch it there, not in submit_execution's namespace
    (this tool has no get_ibm_service call of its own).

    _transpile is patched for the same reason _make_sampler is: it is a thin lazy
    wrapper around qiskit.transpile, and the SimpleNamespace sdk_backend these
    tests use is not a BackendV2 (a real fake such as GenericBackendV2 is no help
    either -- it has no .properties(), which build_hardware_graph needs).

    Yields (make_sampler_mock, transpile_mock) so every test can assert on -- or
    assert the absence of -- those calls.
    """
    with (
        patch(f"{_MAP_MODULE}.get_ibm_service", return_value=service),
        patch(f"{_MAP_MODULE}.build_hardware_graph", return_value=hw_graph or _fake_hw_graph()),
        patch(f"{_SUBMIT_MODULE}._make_sampler", return_value=sampler) as make_sampler,
        patch(f"{_SUBMIT_MODULE}._transpile", return_value=isa_circuit) as transpile,
    ):
        yield make_sampler, transpile


# --------------------------------------------------------------------------
# Cap enforcement: a rejected request must never reach anything networked.
# --------------------------------------------------------------------------


def test_ibm_submit_execution_rejects_missing_backend():
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    service, _resolved = _fake_service()
    with _patched_sdk(service) as (make_sampler, transpile), pytest.raises(ValidationError):
        IbmSubmitExecutionInput(circuit_preset="bell", shots=100)  # backend omitted

    make_sampler.assert_not_called()
    transpile.assert_not_called()


def test_ibm_submit_execution_rejects_too_many_shots(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    service, _resolved = _fake_service()
    with _patched_sdk(service) as (make_sampler, transpile):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=5000)
        )

    assert out.ok is False
    assert "shots" in out.error_message.lower()
    # A capped-out request must not transpile or build a sampler -- nothing
    # network-adjacent, nothing that could spend credits.
    make_sampler.assert_not_called()
    transpile.assert_not_called()


def test_ibm_submit_execution_rejects_too_many_qubits(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    service, _resolved = _fake_service()
    with _patched_sdk(service) as (make_sampler, transpile):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(
                circuit_preset="ghz", num_qubits=11, backend="ibm_fake", shots=100
            )
        )

    assert out.ok is False
    assert "qubit" in out.error_message.lower()
    make_sampler.assert_not_called()
    transpile.assert_not_called()


def test_ibm_submit_execution_qaoa_layers_out_of_bounds_rejected():
    """qaoa_layers must be bounded the same way Task 12 bounded it on
    IbmPreviewExecutionInput -- unbounded qaoa_layers is a cheap-to-request,
    expensive-to-build hazard (compute_mapping's construction is O(n * layers))
    independent of the credit-spend caps this tool otherwise enforces at
    runtime."""
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(backend="ibm_fake", shots=100, qaoa_layers=0)

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(backend="ibm_fake", shots=100, qaoa_layers=1001)


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_ibm_submit_execution_rejects_non_positive_num_qubits(num_qubits):
    """num_qubits <= 0 must be a ValidationError at construction, not something
    that reaches qiskit and surfaces as a bare CircuitError: the runtime cap
    check below only guards the upper bound, so without ge=1 a zero/negative
    request sailed straight into circuit construction."""
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(
            circuit_preset="ghz", num_qubits=num_qubits, backend="ibm_fake", shots=100
        )


@pytest.mark.parametrize("shots", [0, -1])
def test_ibm_submit_execution_rejects_non_positive_shots(shots):
    """Same reasoning as num_qubits: the runtime cap only checks shots > 4000."""
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
    )

    with pytest.raises(ValidationError):
        IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=shots)


# --------------------------------------------------------------------------
# The circuit that actually gets submitted.
# --------------------------------------------------------------------------


def test_ibm_submit_execution_submits_the_mapped_circuit_not_an_empty_pub_list(
    monkeypatch, tmp_path
):
    """CORE REGRESSION TEST.

    The tool used to call sampler.run([], shots=...) -- an empty PUB list. That
    spends real IBM Quantum credits on a job containing nothing, and reports
    success while never running the user's circuit. sampler.run must receive a
    one-element list holding exactly the ISA circuit _transpile produced.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    isa_circuit = SimpleNamespace(tag="isa-circuit-sentinel")
    sampler = MagicMock()
    sampler.run.return_value = _fake_job()
    service, _resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=isa_circuit) as (_, transpile):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    transpile.assert_called_once()
    sampler.run.assert_called_once()

    pubs = sampler.run.call_args.args[0]
    assert pubs != [], "sampler.run was handed an empty PUB list -- no circuit would run"
    assert len(pubs) == 1
    assert pubs[0] is isa_circuit
    assert pubs == [isa_circuit]
    assert sampler.run.call_args.kwargs == {"shots": 100}


def test_ibm_submit_execution_transpiles_with_measurements_and_the_resolved_placement(
    monkeypatch, tmp_path
):
    """The circuit handed to _transpile must be runnable: measurements added (the
    mapping/preview path never needs them, so the preset circuits arrive with
    none) and the resolved logical->physical placement passed as initial_layout,
    against the very backend compute_mapping already resolved."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import compute_mapping
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    sampler = MagicMock()
    sampler.run.return_value = _fake_job()
    service, resolved = _fake_service()

    # A skewed graph so the placement is non-identity -- see _skewed_hw_graph.
    hw_graph = _skewed_hw_graph()

    with _patched_sdk(
        service, sampler=sampler, isa_circuit=SimpleNamespace(), hw_graph=hw_graph
    ) as (_, transpile):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )
        sdk_backend = resolved[0]
        # Same patches, same inputs -> the same deterministic placement the tool
        # itself resolved, so initial_layout can be asserted exactly.
        expected = compute_mapping(
            circuit_qasm=None,
            circuit_preset="bell",
            num_qubits=None,
            qaoa_layers=None,
            backend="ibm_fake",
            objective="maximum_fidelity",
            include_svg=False,
        )

    assert out.ok is True
    assert expected.placement is not None
    assert not expected.circuit.clbits, "preset circuits arrive without measurements"
    expected_layout = [expected.placement.assignment[i] for i in range(2)]
    assert expected_layout != [0, 1], "test setup: placement must be non-identity to be meaningful"

    circuit_arg, backend_arg, layout_arg = transpile.call_args.args
    assert circuit_arg.num_qubits == 2
    assert circuit_arg.clbits, "measurements were not added before submission"
    assert backend_arg is sdk_backend
    assert layout_arg == expected_layout


def test_ibm_submit_execution_reuses_the_mapping_sdk_backend_for_the_sampler(monkeypatch, tmp_path):
    """_make_sampler must be handed compute_mapping's already-authenticated
    service and the very sdk_backend object it resolved -- identity, not just
    equality -- so this tool never re-authenticates or re-fetches the backend."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    sampler = MagicMock()
    sampler.run.return_value = _fake_job()
    service, resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=SimpleNamespace()) as (
        make_sampler,
        transpile,
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    # Exactly one service.backend() call for the whole submission -- nothing
    # re-fetched the backend after compute_mapping had already resolved it.
    assert len(resolved) == 1
    make_sampler.assert_called_once()
    service_arg, backend_arg = make_sampler.call_args.args
    assert service_arg is service
    assert backend_arg is resolved[0]
    # _transpile targets the same resolved backend object.
    assert transpile.call_args.args[1] is resolved[0]


def test_ibm_submit_execution_success(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    isa_circuit = SimpleNamespace(tag="isa-circuit-sentinel")
    sampler = SimpleNamespace(run=lambda *a, **kw: _fake_job())
    service, _resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=isa_circuit):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    assert out.job_id == "job-123"
    assert out.status == "QUEUED"
    assert out.backend == "ibm_fake"  # the resolved backend name
    assert out.shots == 100
    assert out.error_message == ""


# --------------------------------------------------------------------------
# Failure handling around the credit-spending call.
# --------------------------------------------------------------------------


def test_ibm_submit_execution_handles_sampler_run_raising(monkeypatch, tmp_path):
    """sampler.run is a real network operation; if it raises, ibm_submit_execution
    must return ok=False rather than let the exception escape -- and must warn
    that a job may nonetheless exist, since the request may have landed before
    the failure."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    def _boom(*a, **kw):
        raise RuntimeError("network exploded")

    sampler = SimpleNamespace(run=_boom)
    service, _resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=SimpleNamespace()):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert "network exploded" in out.error_message
    assert "a job may still have been created" in out.error_message.lower()
    assert out.job_id == ""


def test_ibm_submit_execution_handles_transpile_raising(monkeypatch, tmp_path):
    """Building the ISA circuit happens before submission; if it fails, the tool
    reports ok=False and never reaches the sampler at all."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    service, _resolved = _fake_service()
    with (
        patch(f"{_MAP_MODULE}.get_ibm_service", return_value=service),
        patch(f"{_MAP_MODULE}.build_hardware_graph", return_value=_fake_hw_graph()),
        patch(f"{_SUBMIT_MODULE}._make_sampler") as make_sampler,
        patch(f"{_SUBMIT_MODULE}._transpile", side_effect=RuntimeError("transpile exploded")),
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert "transpile exploded" in out.error_message
    make_sampler.assert_not_called()


def test_ibm_submit_execution_preserves_job_id_when_status_lookup_fails(monkeypatch, tmp_path):
    """Once sampler.run has returned, the job is real and already spending
    credits. A failure in the follow-up job.status() lookup must NOT discard the
    job_id we already hold, and must not flip the result to ok=False -- otherwise
    the caller loses the only handle to a live job.

    (job.job_id() is a plain local attribute read in the real SDK and cannot
    raise, so it is not what this test simulates; status() is the call that
    round-trips to the service.)"""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    def _boom():
        raise RuntimeError("status lookup exploded")

    job = SimpleNamespace(job_id=lambda: "job-123", status=_boom)
    sampler = SimpleNamespace(run=lambda *a, **kw: job)
    service, _resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=SimpleNamespace()):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is True
    assert out.job_id == "job-123"
    assert "status lookup failed" in out.status
    assert "unknown" in out.status
    assert "status lookup exploded" in out.status
    assert out.error_message == ""


# --------------------------------------------------------------------------
# A 0-qubit / empty-placement request must never reach the credit spend.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qasm",
    ["", "   ", 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[0];\n'],
    ids=["empty", "whitespace", "zero-width-qreg"],
)
def test_ibm_submit_execution_rejects_zero_qubit_circuit(qasm, monkeypatch, tmp_path):
    """CORE REGRESSION TEST.

    QuantumCircuit.from_qasm_str("") returns a valid 0-qubit circuit rather than
    raising, so an empty circuit_qasm used to sail through parse_circuit_input,
    produce an empty placement (which passes every size/cap check trivially) and
    reach sampler.run -- spending real account credits on a job that runs
    nothing, while reporting ok=True with a job_id as if it had worked.

    Nothing network-adjacent may be touched: not _transpile, not _make_sampler,
    not sampler.run.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
        IbmSubmitExecutionInput,
        ibm_submit_execution,
    )

    sampler = MagicMock()
    sampler.run.return_value = _fake_job()
    service, _resolved = _fake_service()

    with _patched_sdk(service, sampler=sampler, isa_circuit=SimpleNamespace()) as (
        make_sampler,
        transpile,
    ):
        out = ibm_submit_execution(
            IbmSubmitExecutionInput(circuit_qasm=qasm, backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert out.job_id == ""
    assert "0 qubits" in out.error_message
    transpile.assert_not_called()
    make_sampler.assert_not_called()
    sampler.run.assert_not_called()


def test_ibm_submit_execution_refuses_an_empty_placement(monkeypatch, tmp_path):
    """Belt-and-braces guard in the one credit-spending tool, independent of
    the upstream 0-qubit rejection: if compute_mapping ever returns an
    ok=True outcome with an empty placement, submission must still be refused
    rather than billing the account for a job with no qubits."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from chatqec_mcp_tools.tools.ibm_hardware import submit_execution as se
    from chatqec_mcp_tools.tools.ibm_hardware._placement import PlacementResult
    from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import MappingOutcome

    sampler = MagicMock()
    sampler.run.return_value = _fake_job()

    empty_outcome = MappingOutcome(
        ok=True,
        backend="ibm_fake",
        placement=PlacementResult(assignment={}, strategy_used="greedy"),
        circuit=SimpleNamespace(clbits=[], num_qubits=0),
        sdk_backend=SimpleNamespace(name="ibm_fake"),
        service=SimpleNamespace(),
    )

    with (
        patch(f"{_SUBMIT_MODULE}.compute_mapping", return_value=empty_outcome),
        patch(f"{_SUBMIT_MODULE}._make_sampler", return_value=sampler) as make_sampler,
        patch(f"{_SUBMIT_MODULE}._transpile") as transpile,
    ):
        out = se.ibm_submit_execution(
            se.IbmSubmitExecutionInput(circuit_preset="bell", backend="ibm_fake", shots=100)
        )

    assert out.ok is False
    assert out.job_id == ""
    assert "empty" in out.error_message.lower()
    transpile.assert_not_called()
    make_sampler.assert_not_called()
    sampler.run.assert_not_called()
