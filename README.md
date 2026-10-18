# MazePy

Reusable geometry foundations for spatial navigation and neuroscience.

This development version implements **only `mazepy.geometry`**. The agreed
repository architecture is retained; all other application modules contain only
empty `.gitkeep` placeholders, so Git preserves their directories. This is a
clean scaffold for the new architecture, rather than an in-place migration of
the original MazePy or MazeProject code.

## Install

Python 3.10 or newer is required. From this repository directory:

```bash
python -m pip install -e .
```

For development and verification:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The geometry implementation needs NumPy and Shapely. No GUI, GPU, SciPy,
simulation, RL, or recording-format dependency is required.

## First example

```python
from mazepy.geometry import Arena2D

arena = Arena2D(
    boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
    walls=[(5, 0, 5, 8)],
    unit="cm",
)

start = (4, 1)
goal = (6, 1)

assert not arena.segment_is_valid(start, goal)
assert arena.distance(start, goal, metric="euclidean") == 2.0

path = arena.shortest_path(start, goal)
print(path.points)
# [[4. 1.]
#  [5. 8.]
#  [6. 1.]]

print(path.length)  # 14.142135623730951 cm

# Sample the geometric path while preserving corners; no motion model is used.
samples = path.sample(spacing=0.5)
assert arena.path_is_valid(samples)
```

`walls` accepts `(n, 4)` segments `(x1, y1, x2, y2)` or `(n, 2, 2)` endpoint
pairs. Empty walls are supported. The outer polygon can be concave, and walls
can be horizontal, vertical, or diagonal.

Solid obstacles, including finite-thickness barriers, use polygon vertices:

```python
arena = Arena2D(
    boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
    obstacles=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
)
path = arena.shortest_path((2, 5), (8, 5))
```

## Public API

| Object or method | Behavior |
| --- | --- |
| `Arena2D(boundary, walls=None, obstacles=None, unit="cm", tolerance=1e-9)` | Validate and retain immutable physical geometry |
| `arena.contains(points, include_boundary=False)` | One boolean or a batch mask; nonfinite positions return False |
| `arena.segment_is_valid(start, end, allow_touch=True)` | Check the complete straight movement segment |
| `arena.path_is_valid(points, allow_touch=True)` | Validate a polyline and its transitions at wall junctions |
| `arena.shortest_path(start, end)` | Return a `Path`; raise `NoPathError` if disconnected |
| `arena.distance(start, end, metric="geodesic")` | Geodesic distance, or infinity for disconnection |
| `arena.distance(start, end, metric="euclidean")` | Straight-line distance, ignoring intervening walls |
| `arena.with_wall(segment)` / `arena.without_wall(index)` | Return new arenas without altering existing geometry or caches |
| `arena.boundary`, `arena.walls`, `arena.obstacles` | Read-only coordinate arrays |
| `arena.area`, `arena.bounds`, `arena.unit`, `arena.fingerprint` | Geometry properties and provenance |
| `PathPlanner(arena, cache_size=128)` | Reusable planner with bounded endpoint-visibility caching |
| `planner.distances(starts, ends)` | Pairwise distance queries; a singleton endpoint broadcasts |
| `planner.graph` / `build_visibility_graph(arena)` | Inspect geometric corner nodes and weighted edges |
| `planner.cache_info()` / `planner.clear_cache()` | Inspect caches or clear endpoint entries |
| `path.points`, `path.length`, `path.unit`, `path.arena_fingerprint` | Geometric path result and provenance |
| `path.sample(spacing)` | Subdivide segments, retaining all corners and endpoints |

Batch distance queries return one distance per pair, rather than an all-pairs
distance matrix. Inputs with invalid shapes raise `GeometryError`. Path queries
with invalid endpoints raise `InvalidPointError`; those errors are never
converted into infinity.

## Contact and numerical conventions

- Coordinates are `(x, y)`, in one explicitly labeled unit. No unit conversion
  or camera transformation is performed.
- Default containment and all path endpoints require strict free-space
  membership. Points on polygon boundaries or on internal walls are excluded.
  Boundary and internal-wall membership use the arena's absolute coordinate
  tolerance.
- With `include_boundary=True`, containment includes permitted boundary/wall
  contact within tolerance. This does not imply connectivity across a wall.
- Collision checks default to an **ideal point agent with boundary contact
  allowed**. Paths can touch wall tips or follow boundary edges. They cannot
  cross walls, switch sides through sealed junctions, or leave the arena.
- `allow_touch=False` rejects all contact with polygon boundaries and internal
  walls. Shortest paths use the contact-allowed model; an obstacle-avoiding
  minimum need not be attained if every boundary contact is forbidden.
- Intersections and overlapping wall segments are noded before topology is
  analyzed. Local free-space sectors prevent shortcuts through T junctions,
  crossings, boundary attachments, and point contacts between obstacles.
- `tolerance` classifies numerical contacts. If a wall endpoint falls just
  outside a polygon edge through rounding, nearby polygon rings are reconciled
  with that endpoint within tolerance. The stored boundary reports that
  normalized geometry. Resolve features well above tolerance and floating-point
  precision; tiny gaps below tolerance are not a supported modeling scale.
- Automatic animal-radius inflation and curved boundaries are deferred. Express
  solid walls as polygons when their physical thickness matters.

Geodesics are shortest Euclidean-length polylines in this polygonal point-agent
model, subject to floating-point geometry precision. They are not routes along
bin centers and are not trajectories satisfying acceleration or turn limits.

## Implementation

Shapely validates polygons, unions solid obstacles, nodes barrier intersections,
and supplies a spatial index. Visibility nodes are reflex free-space corners
and free wall tips; shortest paths do not need to bend at convex free-space
corners. Incident angular sectors and continuous-barrier side checks preserve
connectivity for zero-thickness walls.

The static corner graph is built lazily and reused. Each query connects its
endpoints to visible corners and searches weighted adjacency lists with
Dijkstra's algorithm. No full graph is copied per query. Straight unobstructed
queries skip graph construction. Geometry edits return new arenas, keeping
fingerprints and derived caches consistent.

Graph construction has a quadratic number of candidate corner pairs. This
version is intended for modest, static polygonal mazes. Large maps and changing
walls should be benchmarked before a real-time workload is assumed.

## Verification snapshot

The implementation passed 82 tests, including analytic shortest-path distances,
wall junctions, disconnected regions, rotation/translation/scale invariance,
sampled obstacle-edge paths, and comparison against independent flood fill in
eight randomly generated square-bin mazes. Ruff lint and formatting checks pass.

Verification used Linux, Python 3.12, NumPy 2.3.5, and Shapely 2.2.0. Windows has
not been exercised in this environment.

An illustrative 12-by-12 perfect-maze workload (random seed 20261018) contained
121 internal wall segments, 70 turning nodes, and 181 visibility edges. Static
graph construction took about 0.46 seconds; 100 random continuous path queries
took about 1.65 seconds in total, with all resulting polylines validated. These
are measurements from one execution environment, not performance guarantees.

## Repository architecture

| Path under `src/mazepy/` | Current state |
| --- | --- |
| `geometry/` | Implemented and tested |
| `environments/` | Empty placeholder |
| `spatial/` | Empty placeholder |
| `data/` | Empty placeholder |
| `behavior/` | Empty placeholder |
| `neural/` | Empty placeholder |
| `simulation/` | Empty placeholder |
| `rl/` | Empty placeholder |
| `plotting/` | Empty placeholder |
| `io/` | Empty placeholder |

The root also retains `tests/`, `docs/`, `examples/`, `benchmarks/`, and
`.github/workflows/`. Only geometry tests are implemented. The other support
directories contain `.gitkeep` files. Geometry imports no future module.

The wheel intentionally installs only the implemented package; the source
archive retains the complete repository scaffold. No compatibility claim is
made for imports from the earlier MazePy version.
