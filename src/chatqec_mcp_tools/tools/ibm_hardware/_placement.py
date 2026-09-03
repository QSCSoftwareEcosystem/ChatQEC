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
        seed = max(hw_graph.nodes, key=quality.node_score)
        assignment[first] = seed
        used_physical.add(seed)

        for logical in order[1:]:
            candidates = _physical_candidates(hw_graph, interaction_graph, logical, assignment, used_physical)
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

    remaining_logical = [n for n in interaction_graph.nodes if n not in assignment]
    remaining_physical = sorted(set(hw_graph.nodes) - used_physical, key=lambda q: -quality.node_score(q))
    for logical, phys in zip(remaining_logical, remaining_physical, strict=False):
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
