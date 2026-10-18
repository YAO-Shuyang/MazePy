"""Validated arena geometry, collision checks, and continuous geodesic paths."""

from .arena import Arena2D
from .collision import CollisionChecker
from .exceptions import GeometryError, InvalidPointError, NoPathError
from .pathfinding import Path, PathPlanner
from .visibility import VisibilityGraph, build_visibility_graph

__all__ = [
    "Arena2D",
    "CollisionChecker",
    "GeometryError",
    "InvalidPointError",
    "NoPathError",
    "Path",
    "PathPlanner",
    "VisibilityGraph",
    "build_visibility_graph",
]
