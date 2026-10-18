"""Cached continuous shortest paths without per-query graph copies."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._topology import XY
from .exceptions import GeometryError, InvalidPointError, NoPathError
from .validation import as_point, as_points, immutable_array
from .visibility import VisibilityGraph, _build_visibility_graph

if TYPE_CHECKING:
    from .arena import Arena2D


@dataclass(frozen=True, slots=True, eq=False)
class Path:
    """A piecewise-linear geodesic, with immutable (x, y) waypoints.

    Waypoints may touch polygon boundaries or walls, as permitted for an ideal
    point agent. This geometric path contains no motion dynamics or timestamps.
    """

    points: NDArray[np.float64] = field(repr=False)
    unit: str
    arena_fingerprint: str
    length: float = field(init=False)

    def __post_init__(self) -> None:
        points, scalar = as_points(self.points)
        if scalar or len(points) == 0:
            raise GeometryError("Path points must have shape (n, 2), with n >= 1.")
        object.__setattr__(self, "points", immutable_array(points))
        object.__setattr__(
            self, "length", float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        )

    def sample(self, spacing: float) -> NDArray[np.float64]:
        """Sample with maximum point spacing, retaining every original corner.

        Preserving corners ensures that joining consecutive output samples does
        not cut through a wall. This performs no curve fitting or smoothing.
        """
        if not np.isfinite(spacing) or spacing <= 0:
            raise GeometryError("spacing must be a finite positive number.")
        pieces = [self.points[:1]]
        for start, end in zip(self.points, self.points[1:], strict=False):
            count = max(1, math.ceil(math.dist(start, end) / spacing))
            pieces.append(np.linspace(start, end, count + 1)[1:])
        return np.concatenate(pieces)


class PathPlanner:
    """Continuous Euclidean-length geodesics in one immutable arena.

    The corner visibility graph is built lazily and reused. Query endpoints are
    connected using temporary adjacency lists; the static graph is never copied.
    A bounded LRU cache retains endpoint visibility for repeated queries.
    """

    def __init__(self, arena: Arena2D, *, cache_size: int = 128) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int):
            raise GeometryError("cache_size must be a nonnegative integer.")
        if cache_size < 0:
            raise GeometryError("cache_size must be a nonnegative integer.")
        self.arena = arena
        self.cache_size = cache_size
        self._checker = arena.collision_checker
        self._graph: VisibilityGraph | None = None
        self._link_cache: OrderedDict[XY, tuple[tuple[int, float], ...]] = OrderedDict()

    @property
    def graph(self) -> VisibilityGraph:
        """Build once on first access; return the immutable static corner graph."""
        if self._graph is None:
            self._graph = _build_visibility_graph(self._checker)
        return self._graph

    def _validate_endpoint(self, values: ArrayLike, name: str) -> XY:
        point = as_point(values, name=name)
        if not self.arena.contains(point):
            raise InvalidPointError(
                f"{name} {point} must be strictly inside free space, "
                "away from walls, obstacles, and the outer boundary."
            )
        return point

    def _links(self, point: XY) -> tuple[tuple[int, float], ...]:
        if point in self._link_cache:
            self._link_cache.move_to_end(point)
            return self._link_cache[point]
        graph = self.graph
        result = tuple(
            (i, math.dist(point, corner))
            for i, corner in enumerate(graph.points)
            if self._checker._connectable(point, tuple(corner), None, graph._sectors[i])
        )
        if self.cache_size:
            self._link_cache[point] = result
            if len(self._link_cache) > self.cache_size:
                self._link_cache.popitem(last=False)
        return result

    def shortest_path(self, start: ArrayLike, end: ArrayLike) -> Path:
        """Return a shortest path; invalid endpoints and disconnection are distinct."""
        p = self._validate_endpoint(start, "start")
        q = self._validate_endpoint(end, "end")
        if p == q:
            return self._result([p])
        if self._checker._connectable(p, q):
            return self._result([p, q])
        start_links = self._links(p)
        goal_links = dict(self._links(q))
        graph = self.graph
        distances = [math.inf] * graph.n_nodes
        parents = [-1] * graph.n_nodes
        queue = []
        for node, distance in start_links:
            distances[node] = distance
            heapq.heappush(queue, (distance, node))
        best = math.inf
        terminal = -1
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            if distance >= best:
                break
            candidate = distance + goal_links.get(node, math.inf)
            if candidate < best:
                best, terminal = candidate, node
            for neighbor, weight in graph.adjacency[node]:
                candidate = distance + weight
                if candidate < distances[neighbor] and candidate < best:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        if terminal < 0:
            raise NoPathError(f"No navigable path connects {p} and {q}.")
        indices = []
        while terminal >= 0:
            indices.append(terminal)
            terminal = parents[terminal]
        corners = [tuple(graph.points[i]) for i in reversed(indices)]
        return self._result([p, *corners, q])

    def _result(self, points: list[XY]) -> Path:
        return Path(np.asarray(points), self.arena.unit, self.arena.fingerprint)

    def distance(self, start: ArrayLike, end: ArrayLike) -> float:
        """Return geodesic distance; disconnected endpoints return infinity."""
        try:
            return self.shortest_path(start, end).length
        except NoPathError:
            return math.inf

    def distances(self, starts: ArrayLike, ends: ArrayLike) -> NDArray[np.float64]:
        """Pairwise queries, broadcasting one endpoint over an (n, 2) array.

        This returns n distances, not an n-by-n matrix. Any malformed or invalid
        endpoint raises; disconnection alone yields infinity.
        """
        start_array, _ = as_points(starts, name="starts")
        end_array, _ = as_points(ends, name="ends")
        n_start, n_end = len(start_array), len(end_array)
        if n_start == n_end:
            return np.asarray(
                [
                    self.distance(a, b)
                    for a, b in zip(start_array, end_array, strict=True)
                ]
            )
        if n_start == 1:
            return np.asarray([self.distance(start_array[0], b) for b in end_array])
        if n_end == 1:
            return np.asarray([self.distance(a, end_array[0]) for a in start_array])
        raise GeometryError("starts and ends need equal lengths or one singleton.")

    def clear_cache(self) -> None:
        """Clear endpoint-visibility entries, retaining the static graph."""
        self._link_cache.clear()

    def cache_info(self) -> dict[str, int | bool]:
        """Inspect cache sizes without triggering graph construction."""
        return {
            "graph_built": self._graph is not None,
            "nodes": self._graph.n_nodes if self._graph is not None else 0,
            "edges": self._graph.n_edges if self._graph is not None else 0,
            "endpoint_entries": len(self._link_cache),
            "endpoint_capacity": self.cache_size,
        }
