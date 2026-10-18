"""Immutable, validated descriptions of planar physical environments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .exceptions import GeometryError
from .validation import as_points, as_ring, as_walls, immutable_array

if TYPE_CHECKING:
    from .collision import CollisionChecker
    from .pathfinding import Path, PathPlanner


def _line_segments(geometry: BaseGeometry) -> list[LineString]:
    """Split noded linework at every coordinate, including bends."""
    if geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        coords = list(geometry.coords)
        return [
            LineString([a, b])
            for a, b in zip(coords, coords[1:], strict=False)
            if a != b
        ]
    result: list[LineString] = []
    for part in geometry.geoms:
        result.extend(_line_segments(part))
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Arena2D:
    """A polygonal arena with zero-thickness walls and solid obstacles.

    Parameters
    ----------
    boundary : array-like, shape (n_vertices, 2)
        Ordered vertices of one simple outer polygon. Either winding is valid.
    walls : array-like, optional
        Segments of shape (n_walls, 4) or (n_walls, 2, 2). Junctions, overlaps,
        diagonal segments, and walls meeting the outer boundary are supported.
    obstacles : sequence of array-like, optional
        Ordered polygon vertices for solid obstacles. Obstacles may touch or
        overlap; their union is removed from the outer polygon.
    unit : str, default "cm"
        A coordinate-unit label; coordinates are never silently converted.
    tolerance : float, default 1e-9
        Absolute coordinate tolerance for contact classification. Near-boundary
        wall endpoints affected by rounding are reconciled within this tolerance;
        walls are not inflated. Resolve features well above this tolerance and
        floating-point precision.

    Notes
    -----
    Inputs are copied into immutable storage. Geometry edits return new arenas.
    Navigation uses an ideal point agent; finite-width barriers can be expressed
    as obstacle polygons. Path endpoints must lie strictly in free space.
    """

    _boundary: NDArray[np.float64] = field(repr=False)
    _walls: NDArray[np.float64] = field(repr=False)
    _obstacles: tuple[NDArray[np.float64], ...] = field(repr=False)
    unit: str
    tolerance: float
    fingerprint: str
    _domain: Polygon | MultiPolygon = field(repr=False)
    _wall_geometry: BaseGeometry = field(repr=False)
    _segments: tuple[LineString, ...] = field(repr=False)
    _tree: STRtree = field(repr=False)
    _checker: CollisionChecker | None = field(default=None, repr=False)
    _planner: PathPlanner | None = field(default=None, repr=False)

    def __init__(
        self,
        boundary: ArrayLike,
        walls: ArrayLike | None = None,
        obstacles: Sequence[ArrayLike] | None = None,
        *,
        unit: str = "cm",
        tolerance: float = 1e-9,
    ) -> None:
        if not isinstance(unit, str) or not unit.strip():
            raise GeometryError("unit must be a nonempty string.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise GeometryError("tolerance must be a finite positive number.")
        tolerance = float(tolerance)
        outer = as_ring(boundary, name="boundary", tolerance=tolerance)
        rings = [] if obstacles is None else list(obstacles)
        solids = [
            as_ring(ring, name=f"obstacle {i}", tolerance=tolerance)
            for i, ring in enumerate(rings)
        ]
        for i, solid in enumerate(solids):
            if not outer.covers(solid):
                raise GeometryError(f"Obstacle {i} extends outside the boundary.")
        domain = outer.difference(shapely.union_all(solids)) if solids else outer
        if (
            domain.is_empty
            or domain.area <= tolerance**2
            or not isinstance(domain, (Polygon, MultiPolygon))
            or not domain.is_valid
        ):
            raise GeometryError("The obstacles leave no valid polygonal free space.")
        wall_array = as_walls(walls, tolerance=tolerance)
        wall_lines = [LineString(row.reshape(2, 2)) for row in wall_array]
        if wall_lines and any(not domain.covers(line) for line in wall_lines):
            # A rotated wall endpoint on an edge can fall a few ulps outside.
            # Insert its exact coordinate into nearby rings so the wall and
            # boundary share a node, instead of inventing a microscopic gap.
            reference = MultiPoint(wall_array.reshape(-1, 2))
            outer = shapely.snap(outer, reference, tolerance)
            solids = [shapely.snap(solid, reference, tolerance) for solid in solids]
            domain = outer.difference(shapely.union_all(solids)) if solids else outer
            if not domain.is_valid or domain.is_empty:
                raise GeometryError("Boundary contacts cannot be normalized safely.")
        for i, line in enumerate(wall_lines):
            if not domain.covers(line):
                raise GeometryError(
                    f"Wall {i} leaves the arena or crosses a solid obstacle."
                )
        wall_geometry = (
            shapely.union_all(wall_lines) if wall_lines else MultiLineString([])
        )
        network = shapely.union_all([domain.boundary, *wall_lines])
        segments = tuple(_line_segments(network))
        digest = hashlib.sha256()
        digest.update(domain.normalize().wkb)
        digest.update(wall_geometry.normalize().wkb)
        digest.update(json.dumps([unit, tolerance]).encode("utf-8"))
        object.__setattr__(
            self, "_boundary", immutable_array(outer.exterior.coords[:-1])
        )
        object.__setattr__(self, "_walls", immutable_array(wall_array))
        object.__setattr__(
            self,
            "_obstacles",
            tuple(immutable_array(solid.exterior.coords[:-1]) for solid in solids),
        )
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "fingerprint", digest.hexdigest())
        object.__setattr__(self, "_domain", domain)
        object.__setattr__(self, "_wall_geometry", wall_geometry)
        object.__setattr__(self, "_segments", segments)
        object.__setattr__(self, "_tree", STRtree(segments))
        object.__setattr__(self, "_checker", None)
        object.__setattr__(self, "_planner", None)

    @property
    def boundary(self) -> NDArray[np.float64]:
        """Outer vertices, without a repeated closing vertex (read-only)."""
        return self._boundary

    @property
    def walls(self) -> NDArray[np.float64]:
        """Input wall segments, shape (n, 4), including an empty (0, 4) array."""
        return self._walls

    @property
    def obstacles(self) -> tuple[NDArray[np.float64], ...]:
        """Read-only input obstacle rings."""
        return self._obstacles

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounds of polygonal free space: (xmin, ymin, xmax, ymax)."""
        return self._domain.bounds

    @property
    def area(self) -> float:
        """Accessible polygonal area; zero-thickness walls remove no area."""
        return float(self._domain.area)

    def contains(
        self, points: ArrayLike, *, include_boundary: bool = False
    ) -> bool | NDArray[np.bool_]:
        """Test positions, returning a bool for one point or an (n,) mask.

        Default: exclude polygon boundaries and points within tolerance of an
        internal wall. With ``include_boundary=True``, allow boundary/wall
        contact. Nonfinite coordinates produce False, making tracking masks
        straightforward. Coordinate shape errors still raise GeometryError.
        """
        coords, scalar = as_points(points, allow_nonfinite=True)
        result = np.zeros(len(coords), dtype=bool)
        finite = np.isfinite(coords).all(axis=1)
        valid_points = shapely.points(coords[finite])
        if include_boundary:
            result[finite] = (
                shapely.distance(self._domain, valid_points) <= self.tolerance
            )
        else:
            inside = shapely.contains(self._domain, valid_points)
            inside &= (
                shapely.distance(self._domain.boundary, valid_points) > self.tolerance
            )
            if not self._wall_geometry.is_empty:
                inside &= (
                    shapely.distance(self._wall_geometry, valid_points) > self.tolerance
                )
            result[finite] = inside
        return bool(result[0]) if scalar else result

    @property
    def collision_checker(self) -> CollisionChecker:
        """Lazily create and reuse a collision checker for this immutable arena."""
        if self._checker is None:
            from .collision import CollisionChecker

            object.__setattr__(self, "_checker", CollisionChecker(self))
        return self._checker

    @property
    def planner(self) -> PathPlanner:
        """Lazily create and reuse the continuous path planner."""
        if self._planner is None:
            from .pathfinding import PathPlanner

            object.__setattr__(self, "_planner", PathPlanner(self))
        return self._planner

    def segment_is_valid(
        self, start: ArrayLike, end: ArrayLike, *, allow_touch: bool = True
    ) -> bool:
        """Check the entire movement segment, including outer-boundary collisions."""
        return self.collision_checker.segment_is_valid(
            start, end, allow_touch=allow_touch
        )

    def path_is_valid(self, points: ArrayLike, *, allow_touch: bool = True) -> bool:
        """Check a polyline, including legal transitions at its corner waypoints."""
        return self.collision_checker.path_is_valid(points, allow_touch=allow_touch)

    def shortest_path(self, start: ArrayLike, end: ArrayLike) -> Path:
        """Return the shortest continuous path or raise NoPathError."""
        return self.planner.shortest_path(start, end)

    def distance(
        self, start: ArrayLike, end: ArrayLike, *, metric: str = "geodesic"
    ) -> float:
        """Return geodesic distance (infinity if disconnected) or Euclidean distance."""
        if metric not in {"geodesic", "euclidean"}:
            raise GeometryError("metric must be 'geodesic' or 'euclidean'.")
        if metric == "euclidean":
            p = self.planner._validate_endpoint(start, "start")
            q = self.planner._validate_endpoint(end, "end")
            return float(np.linalg.norm(np.subtract(p, q)))
        return self.planner.distance(start, end)

    def with_wall(self, wall: ArrayLike) -> Arena2D:
        """Return a new arena with one additional segment; leave this arena intact."""
        row = np.asarray(wall, dtype=np.float64)
        if row.shape not in {(4,), (2, 2)}:
            raise GeometryError("wall must have shape (4,) or (2, 2).")
        return Arena2D(
            self.boundary,
            np.vstack([self.walls, row.reshape(1, 4)]),
            self.obstacles,
            unit=self.unit,
            tolerance=self.tolerance,
        )

    def without_wall(self, index: int) -> Arena2D:
        """Return a new arena without the selected zero-based input wall."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise GeometryError("Wall index must be an integer.")
        if not 0 <= index < len(self.walls):
            raise IndexError("Wall index is out of range.")
        return Arena2D(
            self.boundary,
            np.delete(self.walls, index, axis=0),
            self.obstacles,
            unit=self.unit,
            tolerance=self.tolerance,
        )
