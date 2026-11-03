"""Explicit analytical samples for public geometry metadata and containers."""

import math

import numpy as np
import pytest

from mazepy.geometry import Arena2D, Path, VisibilityGraph, build_visibility_graph


def test_open_square_metadata_example():
    arena = Arena2D([(0, 0), (10, 0), (10, 10), (0, 10)], unit="cm")
    assert arena.bounds == (0.0, 0.0, 10.0, 10.0)
    assert arena.area == pytest.approx(100.0)
    assert arena.collision_checker is arena.collision_checker
    assert arena.planner is arena.planner
    graph = build_visibility_graph(arena)
    assert isinstance(graph, VisibilityGraph)
    assert graph.arena_fingerprint == arena.fingerprint
    for edges in graph.adjacency:
        assert all(weight > 0 for _, weight in edges)
    assert graph.n_nodes >= 0
    assert graph.n_edges >= 0


def test_three_four_five_path_example():
    path = Path(np.array([[0.0, 0.0], [3.0, 4.0]]), "cm", "example")
    assert path.length == pytest.approx(5.0)
    samples = path.sample(1.0)
    np.testing.assert_allclose(samples[0], [0, 0])
    np.testing.assert_allclose(samples[-1], [3, 4])
    sampled_length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
    assert math.isclose(sampled_length, 5.0)
