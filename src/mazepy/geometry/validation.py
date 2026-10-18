"""Input validation shared by geometry objects and queries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from .exceptions import GeometryError


def immutable_array(values: ArrayLike) -> NDArray[np.float64]:
    """Make a detached array whose immutable bytes cannot be made writable."""
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def as_points(
    values: ArrayLike, *, name: str = "points", allow_nonfinite: bool = False
) -> tuple[NDArray[np.float64], bool]:
    """Accept one (x, y) pair or an (n, 2) array; preserve scalar intent."""
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} must contain numeric coordinates.") from exc
    scalar = points.shape == (2,)
    if scalar:
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise GeometryError(f"{name} must have shape (2,) or (n, 2).")
    if not allow_nonfinite and not np.isfinite(points).all():
        raise GeometryError(f"{name} must contain only finite coordinates.")
    return points, scalar


def as_point(values: ArrayLike, *, name: str = "point") -> tuple[float, float]:
    """Validate exactly one coordinate pair."""
    points, scalar = as_points(values, name=name)
    if not scalar:
        raise GeometryError(f"{name} must have shape (2,).")
    return float(points[0, 0]), float(points[0, 1])


def as_ring(values: ArrayLike, *, name: str, tolerance: float) -> Polygon:
    """Validate a simple polygon, accepting an optional repeated last vertex."""
    vertices, scalar = as_points(values, name=name)
    if scalar:
        raise GeometryError(f"{name} needs at least three vertices.")
    if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise GeometryError(f"{name} needs at least three vertices.")
    edges = np.roll(vertices, -1, axis=0) - vertices
    if np.any(np.linalg.norm(edges, axis=1) <= tolerance):
        raise GeometryError(f"{name} has a repeated or too-close adjacent vertex.")
    polygon = Polygon(vertices)
    if not polygon.is_valid:
        raise GeometryError(f"Invalid {name}: {explain_validity(polygon)}")
    if polygon.area <= tolerance**2:
        raise GeometryError(f"{name} must enclose positive area.")
    return polygon


def as_walls(values: ArrayLike | None, *, tolerance: float) -> NDArray[np.float64]:
    """Validate segments supplied as (n, 4) or (n, 2, 2)."""
    if values is None:
        return np.empty((0, 4), dtype=np.float64)
    try:
        walls = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryError("walls must contain numeric coordinates.") from exc
    if walls.size == 0:
        if walls.shape in {(0,), (0, 4), (0, 2, 2)}:
            return np.empty((0, 4), dtype=np.float64)
        raise GeometryError("Empty walls must have shape (0, 4) or (0, 2, 2).")
    if walls.ndim == 3 and walls.shape[1:] == (2, 2):
        walls = walls.reshape(-1, 4)
    if walls.ndim != 2 or walls.shape[1] != 4:
        raise GeometryError("walls must have shape (n, 4) or (n, 2, 2).")
    if not np.isfinite(walls).all():
        raise GeometryError("walls must contain only finite coordinates.")
    if np.any(np.linalg.norm(walls[:, 2:] - walls[:, :2], axis=1) <= tolerance):
        raise GeometryError("Walls must have length greater than tolerance.")
    return walls
