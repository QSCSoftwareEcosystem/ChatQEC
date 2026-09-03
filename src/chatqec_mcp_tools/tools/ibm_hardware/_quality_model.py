"""Objective-based, min-max normalized hardware quality scoring.

Ported essentially as-is from cahgr/retrieval.py's HardwareQualityModel in the
intern's HardwareAwarenessIntegrationPipeline -- the review found this schema
sound. Adjusted to read from this package's HardwareGraph (same attribute
names) and drop the reliability_adjustment feedback-loop parameter (out of
scope here).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
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

        self._t1_norm: Callable[[float], float] = _minmax(t1_vals)
        self._t2_norm: Callable[[float], float] = _minmax(t2_vals)
        self._ro_norm: Callable[[float], float] = _minmax(ro_vals)
        self._fid_norm: Callable[[float], float] = _minmax(fid_vals)
        self._time_norm: Callable[[float], float] = _minmax(time_vals)

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
            with suppress(TypeError, ValueError):
                values.append(float(v))
    return values


def _minmax(values: list[float]) -> Callable[[float], float]:
    if not values:
        return lambda v: 0.5
    lo, hi = min(values), max(values)
    if hi <= lo:
        return lambda v: 1.0

    def _norm(v: float) -> float:
        with suppress(TypeError, ValueError):
            return (float(v) - lo) / (hi - lo)
        return 0.5

    return _norm


__all__ = ["HardwareQualityModel"]
