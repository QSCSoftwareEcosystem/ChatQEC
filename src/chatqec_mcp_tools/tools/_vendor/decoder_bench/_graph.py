"""Minimal stand-in for the part of python-igraph that color_code_gen.py uses.

decoder-bench's color-code generator uses igraph only as an attribute store for its Tanner
graph. igraph is GPL-licensed, so instead of depending on it this module implements exactly the
calls the generator makes, with igraph's semantics:

- ``Graph.add_vertex(**attrs)``, ``Graph.add_edge(i, j)``, ``Graph.vcount()``
- ``Graph.vs``: iterates vertices in index order; ``.select(**filters)`` and ``.find(**filters)``,
  where a filter is ``attr=value``, ``attr_ne=value`` or ``attr_in=values`` (all ANDed),
  ``select`` is chainable and ``find`` raises ValueError when nothing matches
- ``VertexSeq[attr]`` (that attribute's values over the sequence, as a list)
- ``Vertex[attr]``, ``Vertex.index`` and ``Vertex.neighbors()``, in ascending vertex index,
  which is igraph's order for an undirected graph

Parity with igraph is checked by hashing generated circuits against circuits produced by the
original code with igraph installed (tests/test_decoder_bench_vendor.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class Vertex:
    def __init__(self, graph: Graph, index: int) -> None:
        self._graph = graph
        self.index = index

    def __getitem__(self, attr: str) -> Any:
        return self._graph._attrs[self.index].get(attr)

    def neighbors(self) -> list[Vertex]:
        return [Vertex(self._graph, j) for j in sorted(self._graph._adj[self.index])]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Vertex) and other._graph is self._graph and other.index == self.index
        )

    def __hash__(self) -> int:
        return hash((id(self._graph), self.index))


def _matches(attrs: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, want in filters.items():
        if key.endswith("_ne"):
            if attrs.get(key[:-3]) == want:
                return False
        elif key.endswith("_in"):
            if attrs.get(key[:-3]) not in want:
                return False
        elif attrs.get(key) != want:
            return False
    return True


class VertexSeq:
    def __init__(self, graph: Graph, indices: list[int]) -> None:
        self._graph = graph
        self._indices = indices

    def __iter__(self) -> Iterator[Vertex]:
        return (Vertex(self._graph, i) for i in self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, k: int | str) -> Any:
        if isinstance(k, str):  # igraph: seq["attr"] is that attribute's list over the seq
            return [self._graph._attrs[i].get(k) for i in self._indices]
        return Vertex(self._graph, self._indices[k])

    def select(self, **filters: Any) -> VertexSeq:
        attrs = self._graph._attrs
        return VertexSeq(self._graph, [i for i in self._indices if _matches(attrs[i], filters)])

    def find(self, **filters: Any) -> Vertex:
        for i in self._indices:
            if _matches(self._graph._attrs[i], filters):
                return Vertex(self._graph, i)
        raise ValueError(f"no such vertex: {filters}")


class Graph:
    def __init__(self) -> None:
        self._attrs: list[dict[str, Any]] = []
        self._adj: list[list[int]] = []

    def add_vertex(self, **attrs: Any) -> Vertex:
        self._attrs.append(dict(attrs))
        self._adj.append([])
        return Vertex(self, len(self._attrs) - 1)

    def add_edge(self, source: int, target: int) -> None:
        self._adj[source].append(target)
        self._adj[target].append(source)

    def vcount(self) -> int:
        return len(self._attrs)

    @property
    def vs(self) -> VertexSeq:
        return VertexSeq(self, list(range(len(self._attrs))))
