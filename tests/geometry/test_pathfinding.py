import math

import numpy as np
import pytest

from mazepy.geometry import (
    Arena2D,
    GeometryError,
    InvalidPointError,
    NoPathError,
    PathPlanner,
    build_visibility_graph,
)


def test_open_field_is_euclidean_and_does_not_build_graph(open_arena):
    path = open_arena.shortest_path((1, 1), (9, 9))
    assert path.length == pytest.approx(math.sqrt(128))
    assert len(path.points) == 2
    assert open_arena.planner.cache_info()["graph_built"] is False
    assert open_arena.distance((1, 1), (9, 9), metric="euclidean") == path.length


def test_known_single_wall_geodesic(detour_arena):
    path = detour_arena.shortest_path((4, 1), (6, 1))
    assert path.length == pytest.approx(2 * math.sqrt(50))
    np.testing.assert_array_equal(path.points, [(4, 1), (5, 8), (6, 1)])
    assert path.unit == "cm"
    assert path.arena_fingerprint == detour_arena.fingerprint
    assert detour_arena.path_is_valid(path.points)
    assert detour_arena.distance((4, 1), (6, 1), metric="euclidean") == 2


def test_known_rectangle_obstacle_geodesic(boundary):
    arena = Arena2D(boundary, obstacles=[[(4, 4), (6, 4), (6, 6), (4, 6)]])
    path = arena.shortest_path((2, 5), (8, 5))
    assert path.length == pytest.approx(2 * math.sqrt(5) + 2)
    assert arena.path_is_valid(path.points)


@pytest.mark.parametrize(
    "walls, start, end",
    [
        ([(5, 0, 5, 10)], (4, 1), (6, 1)),
        ([(0, 5, 10, 5), (5, 0, 5, 5)], (2, 2), (8, 2)),
        (
            [(3, 3, 7, 3), (7, 3, 7, 7), (7, 7, 3, 7), (3, 7, 3, 3)],
            (5, 5),
            (1, 1),
        ),
    ],
)
def test_disconnected_components(boundary, walls, start, end):
    arena = Arena2D(boundary, walls)
    assert math.isinf(arena.distance(start, end))
    with pytest.raises(NoPathError):
        arena.shortest_path(start, end)


def test_cross_wall_detour_is_legal(boundary):
    arena = Arena2D(boundary, [(5, 2, 5, 8), (2, 5, 8, 5)])
    path = arena.shortest_path((4, 4), (6, 6))
    assert path.length == pytest.approx(2 * math.sqrt(5) + math.sqrt(18))
    assert arena.path_is_valid(path.points)


def test_l_wall_shortcut_is_rejected(boundary):
    arena = Arena2D(boundary, [(2, 2, 8, 2), (2, 2, 2, 8)])
    path = arena.shortest_path((3, 3), (1, 1))
    assert path.length == pytest.approx(math.sqrt(26) + math.sqrt(50))
    assert arena.path_is_valid(path.points)


def test_doorway_and_narrow_gap(boundary):
    arena = Arena2D(boundary, [(5, 0, 5, 4), (5, 6, 5, 10)])
    assert arena.distance((2, 2), (8, 2)) == pytest.approx(2 * math.sqrt(13))
    assert arena.distance((2, 5), (8, 5)) == 6
    narrow = Arena2D(boundary, [(5, 0, 5, 5), (5, 5.0001, 5, 10)])
    assert narrow.distance((2, 5.00005), (8, 5.00005)) == 6


def test_concave_arena_has_known_reflex_corner():
    arena = Arena2D([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    path = arena.shortest_path((3, 0.5), (0.5, 3))
    assert path.length == pytest.approx(2 * math.sqrt(4.25))
    assert arena.path_is_valid(path.points)


@pytest.mark.parametrize("point", [(0, 1), (11, 1), (5, 1), (5, 8)])
def test_invalid_endpoint_is_distinct_from_disconnection(detour_arena, point):
    with pytest.raises(InvalidPointError):
        detour_arena.distance(point, (1, 1))


def test_identical_endpoints(open_arena):
    path = open_arena.shortest_path((1, 1), (1, 1))
    assert path.length == 0
    assert path.points.shape == (1, 2)
    np.testing.assert_array_equal(path.sample(1), [[1, 1]])


def test_sampling_retains_corners_and_maximum_spacing(detour_arena):
    path = detour_arena.shortest_path((4, 1), (6, 1))
    sampled = path.sample(3)
    assert any(np.array_equal(point, [5, 8]) for point in sampled)
    assert np.linalg.norm(np.diff(sampled, axis=0), axis=1).max() <= 3
    assert detour_arena.path_is_valid(sampled)
    assert detour_arena.path_is_valid(path.sample(100))
    with pytest.raises(ValueError):
        path.points.setflags(write=True)
    with pytest.raises(GeometryError):
        path.sample(0)


def test_cache_and_static_graph_are_reused(detour_arena):
    planner = PathPlanner(detour_arena, cache_size=2)
    first = planner.shortest_path((4, 1), (6, 1))
    graph = planner.graph
    assert graph is planner.graph
    assert planner.cache_info()["endpoint_entries"] == 2
    assert planner.shortest_path((4, 1), (6, 1)).length == first.length
    planner.shortest_path((3, 1), (7, 1))
    assert planner.cache_info()["endpoint_entries"] == 2
    assert planner.graph is graph
    planner.clear_cache()
    assert planner.cache_info()["endpoint_entries"] == 0
    assert planner.graph is graph
    with pytest.raises(ValueError):
        graph.points.setflags(write=True)


def test_batched_queries_and_broadcasting(open_arena):
    planner = open_arena.planner
    expected = [1, 2]
    np.testing.assert_allclose(planner.distances((1, 1), [(2, 1), (3, 1)]), expected)
    np.testing.assert_allclose(planner.distances([(2, 1), (3, 1)], (1, 1)), expected)
    np.testing.assert_allclose(
        planner.distances([(1, 1), (1, 1)], [(2, 1), (3, 1)]), expected
    )
    assert planner.distances(np.empty((0, 2)), np.empty((0, 2))).shape == (0,)
    with pytest.raises(GeometryError):
        planner.distances(np.ones((2, 2)), np.ones((3, 2)))


@pytest.mark.parametrize("cache_size", [-1, True, 1.5])
def test_invalid_cache_size(open_arena, cache_size):
    with pytest.raises(GeometryError):
        PathPlanner(open_arena, cache_size=cache_size)


def test_overlapping_and_duplicate_walls(boundary):
    arena = Arena2D(boundary, [(5, 0, 5, 6), (5, 3, 5, 8), (5, 8, 5, 0)])
    assert arena.distance((4, 1), (6, 1)) == pytest.approx(2 * math.sqrt(50))


@pytest.mark.parametrize("scale", [0.001, 1, 1000])
@pytest.mark.parametrize("angle", [0, 0.37, 1.8])
def test_translation_rotation_and_scale_preserve_distance(scale, angle):
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )

    def transform(points):
        return np.asarray(points) @ rotation.T * scale + np.array([13, -21]) * scale

    arena = Arena2D(
        transform([(0, 0), (10, 0), (10, 10), (0, 10)]),
        transform([(5, 0), (5, 8)]).reshape(1, 4),
        tolerance=1e-10 * scale,
    )
    start, end = transform([(4, 1), (6, 1)])
    path = arena.shortest_path(start, end)
    assert path.length == pytest.approx(2 * math.sqrt(50) * scale)
    assert arena.path_is_valid(path.points)


def test_visibility_graph_contains_geometric_corners(detour_arena):
    graph = build_visibility_graph(detour_arena)
    assert graph.n_nodes == 1
    assert graph.n_edges == 0
    np.testing.assert_allclose(graph.points, [[5, 8]])


def test_random_paths_are_symmetric_legal_and_above_straight_line(boundary):
    arena = Arena2D(boundary, [(3, 0, 3, 7), (7, 3, 7, 10)])
    generator = np.random.default_rng(318)
    for _ in range(20):
        start, end = generator.uniform(0.1, 9.9, size=(2, 2))
        path = arena.shortest_path(start, end)
        reverse = arena.shortest_path(end, start)
        assert path.length >= math.dist(start, end) - 1e-10
        assert path.length == pytest.approx(reverse.length)
        assert arena.path_is_valid(path.points)


def test_unknown_metric_is_rejected(open_arena):
    with pytest.raises(GeometryError):
        open_arena.distance((1, 1), (2, 2), metric="graph")
