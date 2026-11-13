"""IQM device snapshot -> the same normalized property graph the IBM tools use.

Nodes are integer indices in the server's qubit order (QB1 -> 0, QB2 -> 1, ...), which is
also the index order of iqm-client's Qiskit backend. Each node keeps its IQM name in
``name``. Attribute names and units match ibm_hardware._hardware_graph (T1/T2 in seconds,
readout_error, gate_error, gate_time in seconds), so HardwareQualityModel, place() and
render_coupling_diagram work unchanged.

T2 here is the echo (Hahn) time, the closest match to the T2 IBM reports; the Ramsey T2*
is kept as ``T2_star`` and shown by iqm_backend_calibration.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import networkx as nx

from chatqec_mcp_tools.tools.ibm_hardware._cache import CacheResult, get_or_fetch
from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import (
    HardwareGraph,
    _fidelity_from_error,
    _is_operational,
)
from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError, call_worker

_DEFAULT_TTL_SECONDS = 600.0


def _cache_key() -> str:
    server = os.environ.get("IQM_SERVER_URL", "")
    qc = os.environ.get("IQM_QUANTUM_COMPUTER", "default")
    digest = hashlib.sha256(f"{server}|{qc}".encode()).hexdigest()[:12]
    return f"iqm_snapshot_{digest}"


def get_snapshot() -> CacheResult:
    """Device snapshot (architecture + quality metrics + gate durations), TTL-cached.

    Raises IQMCredentialsError when credentials or the IQM env are missing; other
    failures fall back to a stale cached snapshot if one exists, else propagate."""
    ttl = float(os.environ.get("CHATQEC_IQM_SNAPSHOT_TTL_SECONDS", _DEFAULT_TTL_SECONDS))
    return get_or_fetch(
        _cache_key(), ttl, lambda: call_worker("snapshot"), reraise=(IQMCredentialsError,)
    )


def _one_minus(fidelity: Any) -> float | None:
    return None if fidelity is None else max(0.0, 1.0 - float(fidelity))


def _readout_error(q: dict[str, Any]) -> float | None:
    fid = q.get("readout_fidelity")
    return None if fid is None else max(0.0, 1.0 - float(fid))


def build_hardware_graph(snapshot: dict[str, Any]) -> HardwareGraph:
    names = [q["name"] for q in snapshot["qubits"]]
    index = {name: i for i, name in enumerate(names)}

    graph = nx.Graph()
    graph.graph["backend_name"] = snapshot.get("dut_label", "")
    for i, q in enumerate(snapshot["qubits"]):
        t1, t2 = q.get("t1_s"), q.get("t2_echo_s")
        readout_error = _readout_error(q)
        graph.add_node(
            i,
            name=q["name"],
            T1=t1,
            T2=t2,
            T2_star=q.get("t2_s"),
            readout_error=readout_error,
            readout_fidelity=_fidelity_from_error(readout_error),
            one_qubit_error=_one_minus(q.get("prx_fidelity")),
            is_operational=_is_operational(t1, t2, readout_error),
        )
    for c in snapshot["couplers"]:
        a, b = c["qubits"]
        if a not in index or b not in index:
            continue
        fid = c.get("cz_fidelity")
        graph.add_edge(
            index[a],
            index[b],
            native_gate="cz",
            gate_error=_one_minus(fid),
            gate_time=c.get("cz_duration_s"),
            gate_fidelity=fid,
        )
    return HardwareGraph(graph=graph, backend_name=snapshot.get("dut_label", ""))


def qubit_names(hw: HardwareGraph) -> list[str]:
    return [hw.graph.nodes[i]["name"] for i in sorted(hw.graph.nodes)]


def named_graph(hw: HardwareGraph) -> nx.Graph:
    """Copy of the graph relabeled QB1, QB2, ... for the coupling diagram."""
    return nx.relabel_nodes(hw.graph, {i: d["name"] for i, d in hw.graph.nodes(data=True)})


__all__ = ["build_hardware_graph", "get_snapshot", "named_graph", "qubit_names"]
