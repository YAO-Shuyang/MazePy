"""Static visibility graphs for shortest continuous paths."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._topology import Sector
from .collision import CollisionChecker
from .validation import immutable_array

if TYPE_CHECKING:
    from .arena import Arena2D


@dataclass(frozen=True, slots=True, eq=False)
class VisibilityGraph:
    """Immutable corner coordinates and weighted adjacency lists.

    Nodes refer to geometric turning corners, not analysis bins. Query endpoints
    are attached separately by PathPlanner, leaving this graph unchanged.
    """

    points: NDArray[np.float64] = field(repr=False)
    adjacency: tuple[tuple[tuple[int, float], ...], ...] = field(repr=False)
    arena_fingerprint: str
    _sectors: tuple[Sector, ...] = field(repr=False)

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return sum(map(len, self.adjacency)) // 2


def _build_visibility_graph(checker: CollisionChecker) -> VisibilityGraph:
    nodes = checker._topology.corner_nodes()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for i, (start, start_sector) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            end, end_sector = nodes[j]
            if checker._connectable(start, end, start_sector, end_sector):
                distance = math.dist(start, end)
                adjacency[i].append((j, distance))
                adjacency[j].append((i, distance))
    return VisibilityGraph(
        points=immutable_array(np.asarray([p for p, _ in nodes]).reshape(-1, 2)),
        adjacency=tuple(tuple(edges) for edges in adjacency),
        arena_fingerprint=checker.arena.fingerprint,
        _sectors=tuple(sector for _, sector in nodes),
    )


def build_visibility_graph(arena: Arena2D) -> VisibilityGraph:
    """Build a reusable visibility graph, respecting wall-junction topology."""
    return _build_visibility_graph(arena.collision_checker)
