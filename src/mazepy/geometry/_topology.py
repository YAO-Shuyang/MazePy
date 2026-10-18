"""Local free-space sectors used to prevent fictitious corner passages.

Zero-thickness walls require more than pairwise intersection tests. At a wall
junction, two separately visible rays may lie in different local free-space
sectors. A path cannot jump between those sectors at a zero-width point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import Point

from .exceptions import GeometryError

if TYPE_CHECKING:
    from .arena import Arena2D

TAU = 2 * math.pi
ANGLE_TOL = 1e-11
LEFT = 1
RIGHT = 2
BOTH = LEFT | RIGHT
XY = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Sector:
    """A connected free angular interval bounded by incident barriers."""

    start: float
    span: float

    def offset(self, angle: float) -> float:
        value = (angle - self.start) % TAU
        return 0.0 if TAU - value < ANGLE_TOL else value

    def contains(self, angle: float) -> bool:
        return self.offset(angle) <= self.span + ANGLE_TOL

    def covers_arc(self, start: float, span: float) -> bool:
        return self.offset(start) + span <= self.span + ANGLE_TOL

    def ray_sides(self, angle: float) -> int:
        """Free sides immediately counterclockwise/clockwise of an outgoing ray."""
        offset = self.offset(angle)
        if offset > self.span + ANGLE_TOL:
            return 0
        if self.span >= TAU - ANGLE_TOL:
            # A dangling wall endpoint has one 2-pi sector cut at its wall ray.
            return BOTH
        result = 0
        if offset < self.span - ANGLE_TOL:
            result |= LEFT
        if offset > ANGLE_TOL:
            result |= RIGHT
        return result


class Topology:
    """Spatial index plus cached local angular topology of an immutable arena."""

    def __init__(self, arena: Arena2D) -> None:
        self.arena = arena
        self._sectors: dict[XY, tuple[Sector, ...]] = {}
        self._segment_array = np.asarray(arena._segments, dtype=object)

    def sectors_at(self, point: XY) -> tuple[Sector, ...]:
        if point in self._sectors:
            return self._sectors[point]
        origin = np.asarray(point)
        distances = shapely.distance(self._segment_array, Point(point))
        incident = distances <= self.arena.tolerance
        angles = []
        radii = list(distances[~incident])
        for segment in self._segment_array[incident]:
            for endpoint in segment.coords:
                vector = np.asarray(endpoint) - origin
                length = float(np.linalg.norm(vector))
                if length > self.arena.tolerance:
                    radii.append(length)
                    angles.append(math.atan2(vector[1], vector[0]) % TAU)
        if not angles:
            return ()
        angles.sort()
        unique = [angles[0]]
        for angle in angles[1:]:
            if angle - unique[-1] > ANGLE_TOL:
                unique.append(angle)
        if len(unique) > 1 and unique[0] + TAU - unique[-1] <= ANGLE_TOL:
            unique.pop()
        radius = min(value for value in radii if value > 0) / 4
        sectors = []
        for i, start in enumerate(unique):
            stop = unique[i + 1] if i + 1 < len(unique) else unique[0] + TAU
            midpoint = (start + stop) / 2
            probe = origin + radius * np.array([math.cos(midpoint), math.sin(midpoint)])
            if np.array_equal(probe, origin):
                raise GeometryError(
                    "Geometry features are below floating-point resolution; "
                    "rescale coordinates or move the origin closer to the arena."
                )
            if self.arena._domain.contains(Point(probe)):
                sectors.append(Sector(start, stop - start))
        result = tuple(sectors)
        self._sectors[point] = result
        return result

    def straight_sides(self, point: XY, angle: float) -> int:
        """Sides on which a straight transit past a barrier contact is possible."""
        result = 0
        for sector in self.sectors_at(point):
            if sector.covers_arc(angle, math.pi):
                result |= LEFT
            if sector.covers_arc(angle + math.pi, math.pi):
                result |= RIGHT
        return result

    def corner_nodes(self) -> tuple[tuple[XY, Sector], ...]:
        """Only free-space reflex corners can bend a shortest path.

        A dangling wall endpoint has a 2-pi sector and is included. Convex
        sectors, T/cross junctions without a reflex sector, and straight wall
        subdivision points cannot shorten a legal path and are omitted.
        """
        vertices = sorted(
            {
                tuple(endpoint)
                for line in self.arena._segments
                for endpoint in line.coords
            }
        )
        nodes = []
        for point in vertices:
            for sector in self.sectors_at(point):
                if sector.span > math.pi + ANGLE_TOL:
                    nodes.append((point, sector))
        return tuple(nodes)
