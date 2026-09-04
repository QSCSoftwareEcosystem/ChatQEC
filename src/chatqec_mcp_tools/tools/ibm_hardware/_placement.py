"""Circuit-to-hardware placement strategies.

place_greedy is ported essentially as-is from cahgr/retrieval.py's
_retrieve_greedy/_physical_candidates -- no defects found there during review.
place_vf2 fixes the one confirmed real defect: the intern's hardcoded
_VF2_NODE_CAP/_VF2_CANDIDATE_CAP had no wall-clock bound. It uses rustworkx's
call_limit to bound the actual number of internal VF2 search states visited
(the real cost driver) -- the same dual-bound pattern (call_limit + a
wall-clock time_limit) Qiskit's own VF2Layout pass uses. A first attempt that
checked time.monotonic() between yields of networkx's
subgraph_monomorphisms_iter() was rejected in review: that check can only run
between yields, so it cannot interrupt VF2's internal backtracking when there
are few or zero matches -- exactly the case that matters most in practice
(no exact match exists) -- and measured ~219s on a pathological 20-node
case. place() tries place_vf2 first (a zero-SWAP result when one exists) and
falls back to place_greedy, which always succeeds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean

import networkx as nx
import rustworkx as rx

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
        seed = max(hw_graph.nodes, key=quality.node_score)
        assignment[first] = seed
        used_physical.add(seed)

        for logical in order[1:]:
            candidates = _physical_candidates(
                hw_graph, interaction_graph, logical, assignment, used_physical
            )
            if not candidates:
                continue

            def score(phys: int, logical: int = logical) -> float:
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


_VF2_MAX_INTERACTION_NODES = (
    20  # cheap pre-filter only; call_limit is the real bound on search cost now
)
_VF2_DEFAULT_TIME_BUDGET_SECONDS = 2.0
_VF2_CALL_LIMIT = (
    100_000  # bounds internal VF2 search states (the real cost driver, per rustworkx's docs)
)


def place_vf2(
    hw_graph: nx.Graph,
    interaction_graph: nx.Graph,
    quality: HardwareQualityModel,
    *,
    time_budget_seconds: float = _VF2_DEFAULT_TIME_BUDGET_SECONDS,
    call_limit: int = _VF2_CALL_LIMIT,
) -> PlacementResult | None:
    """Exact subgraph monomorphism search via rustworkx (bounded by
    call_limit, the count of internal VF2 search states -- not by how many
    already-found candidates get scored). A wall-clock deadline is kept as a
    second, complementary bound (matching Qiskit's own VF2Layout pass, which
    uses both call_limit and time_limit together) in case scoring itself is
    slow for a very large hardware graph. Returns None (never raises) if no
    match is found, the interaction graph is empty/too large, or either bound
    is hit; callers should fall back to place_greedy."""
    if (
        interaction_graph.number_of_nodes() == 0
        or interaction_graph.number_of_nodes() > _VF2_MAX_INTERACTION_NODES
    ):
        return None
    if interaction_graph.number_of_edges() == 0:
        return None  # nothing to align structurally

    hw_rx = rx.networkx_converter(hw_graph, keep_attributes=False)
    inter_rx = rx.networkx_converter(interaction_graph, keep_attributes=False)
    # nx.Graph is always undirected, so networkx_converter always returns a
    # PyGraph here; narrow the PyGraph | PyDiGraph union type for mypy.
    assert isinstance(hw_rx, rx.PyGraph)
    assert isinstance(inter_rx, rx.PyGraph)

    best_assignment: dict[int, int] | None = None
    best_score = float("-inf")
    deadline = time.monotonic() + time_budget_seconds

    mappings = rx.vf2_mapping(hw_rx, inter_rx, subgraph=True, induced=False, call_limit=call_limit)
    for mapping in mappings:
        if time.monotonic() >= deadline:
            break
        # mapping: {hw_rx_index: inter_rx_index}; translate rustworkx indices
        # back to the original nx node labels (payloads), then invert to our
        # logical->physical convention.
        assignment = {inter_rx[inter_idx]: hw_rx[hw_idx] for hw_idx, inter_idx in mapping.items()}
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
    edge_scores: list[float] = []
    for u, v, d in interaction_graph.edges(data=True):
        pu, pv = assignment.get(u), assignment.get(v)
        if pu is None or pv is None:
            continue
        if hw_graph.has_edge(pu, pv):
            edge_scores.append(d["weight"] * quality.edge_score(pu, pv))
    num_edges: int = interaction_graph.number_of_edges()
    edge_component = sum(edge_scores) / max(1, num_edges)
    return 0.3 * node_component + 0.7 * edge_component
