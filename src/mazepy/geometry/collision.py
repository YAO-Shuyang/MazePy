"""Whole-segment and polyline collision checks, including junction topology."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import LineString, Point

from ._topology import BOTH, LEFT, RIGHT, XY, Sector, Topology
from .exceptions import GeometryError
from .validation import as_point, as_points

if TYPE_CHECKING:
    from .arena import Arena2D


def _reverse_sides(mask: int) -> int:
    return (RIGHT if mask & LEFT else 0) | (LEFT if mask & RIGHT else 0)


class CollisionChecker:
    """Reuse an arena's spatial index for collision and local-topology checks.

    With ``allow_touch=True``, boundary following and tangency are allowed in
    the ideal point-agent model. Crossing a wall, squeezing through a sealed
    junction, and leaving the outer polygon are always forbidden. With False,
    every contact with a wall or polygon boundary is forbidden.
    """

    def __init__(self, arena: Arena2D) -> None:
        self.arena = arena
        self._topology = Topology(arena)
        self._contact_domain = arena._domain.buffer(arena.tolerance)

    def segment_is_valid(
        self, start: ArrayLike, end: ArrayLike, *, allow_touch: bool = True
    ) -> bool:
        """Return whether the entire straight segment is traversable."""
        p = as_point(start, name="start")
        q = as_point(end, name="end")
        return self._connectable(p, q, allow_touch=allow_touch)

    def _connectable(
        self,
        start: XY,
        end: XY,
        start_sector: Sector | None = None,
        end_sector: Sector | None = None,
        *,
        allow_touch: bool = True,
    ) -> bool:
        if start == end:
            return bool(self.arena.contains(start, include_boundary=allow_touch))
        delta = np.subtract(end, start)
        length = float(np.linalg.norm(delta))
        angle = math.atan2(delta[1], delta[0])
        if start_sector is not None and not start_sector.contains(angle):
            return False
        if end_sector is not None and not end_sector.contains(angle + math.pi):
            return False
        line = LineString([start, end])
        domain = self._contact_domain if allow_touch else self.arena._domain
        if not domain.covers(line):
            return False
        hits = self.arena._tree.query(
            line.buffer(self.arena.tolerance), predicate="intersects"
        )
        if len(hits) == 0:
            return True
        if not allow_touch:
            return False
        parameter_tol = self.arena.tolerance / length
        contacts: dict[float, XY] = {}
        overlaps: list[tuple[float, float]] = []

        def parameter(point: XY) -> float:
            return float(np.dot(np.subtract(point, start), delta) / length**2)

        for index in hits:
            barrier = self.arena._segments[index]
            a, b = barrier.coords
            unit = delta / length
            # Interpolating a rotated boundary edge can leave it by a few ulps.
            # Treat near-collinear overlap as contact at the declared tolerance.
            offsets = np.asarray([a, b]) - np.asarray(start)
            normal_distances = np.abs(offsets[:, 0] * unit[1] - offsets[:, 1] * unit[0])
            if np.all(normal_distances <= self.arena.tolerance):
                ts = [parameter(a), parameter(b)]
                lo, hi = max(0.0, min(ts)), min(1.0, max(ts))
                if hi - lo > parameter_tol:
                    overlaps.append((lo, hi))
                    for t in (lo, hi):
                        contacts[t] = tuple(np.asarray(start) + t * delta)
                    continue
            intersection = line.intersection(barrier)
            if isinstance(intersection, Point):
                point = (float(intersection.x), float(intersection.y))
                t = parameter(point)
                if parameter_tol < t < 1 - parameter_tol:
                    # A noncollinear interior/interior intersection is a crossing.
                    if min(math.dist(point, a), math.dist(point, b)) > (
                        self.arena.tolerance
                    ):
                        return False
                contacts[t] = point
            elif isinstance(intersection, LineString) and not intersection.is_empty:
                points = [tuple(intersection.coords[0]), tuple(intersection.coords[-1])]
                ts = [parameter(point) for point in points]
                overlaps.append((min(ts), max(ts)))
                contacts.update(zip(ts, points, strict=True))
            elif not intersection.is_empty:
                raise GeometryError("Unexpected intersection in noded barrier network.")

        masks: dict[float, int] = {}
        for t, point in contacts.items():
            if parameter_tol < t < 1 - parameter_tol:
                mask = self._topology.straight_sides(point, angle)
                if mask == 0:
                    return False
                masks[t] = mask

        # Along a continuous barrier, an edge must stay on one physical side.
        # Side changes between isolated tangencies are permitted in open space.
        merged: list[list[float]] = []
        for lo, hi in sorted(overlaps):
            if merged and lo <= merged[-1][1] + parameter_tol:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        for lo, hi in merged:
            mask = BOTH
            for t, contact_mask in masks.items():
                if lo - parameter_tol <= t <= hi + parameter_tol:
                    mask &= contact_mask
            if lo <= parameter_tol and start_sector is not None:
                mask &= start_sector.ray_sides(angle)
            if hi >= 1 - parameter_tol and end_sector is not None:
                mask &= _reverse_sides(end_sector.ray_sides(angle + math.pi))
            if mask == 0:
                return False
        return True

    def path_is_valid(self, points: ArrayLike, *, allow_touch: bool = True) -> bool:
        """Validate a polyline without allowing side changes through wall vertices.

        Checking segments separately is insufficient at waypoints on walls.
        Reachable local sectors are propagated through the complete polyline.
        """
        coords, scalar = as_points(points)
        if scalar or len(coords) == 0:
            raise GeometryError("A path must have shape (n, 2), with n >= 1.")
        if len(coords) > 1:
            keep = np.r_[True, np.any(coords[1:] != coords[:-1], axis=1)]
            coords = coords[keep]
        if len(coords) == 1:
            return bool(self.arena.contains(coords[0], include_boundary=allow_touch))
        if not self.arena.contains(coords, include_boundary=allow_touch).all():
            return False

        def states(point: XY) -> tuple[Sector | None, ...]:
            indices = self.arena._tree.query(Point(point).buffer(self.arena.tolerance))
            if any(
                self.arena._segments[i].distance(Point(point)) <= self.arena.tolerance
                for i in indices
            ):
                return self._topology.sectors_at(point)
            return (None,)

        previous = tuple(coords[0])
        reachable = states(previous)
        for row in coords[1:]:
            current = tuple(row)
            candidates = states(current)
            next_reachable = tuple(
                target
                for target in candidates
                if any(
                    self._connectable(
                        previous, current, source, target, allow_touch=allow_touch
                    )
                    for source in reachable
                )
            )
            if not next_reachable:
                return False
            previous, reachable = current, next_reachable
        return True
