import numpy as np
import pytest

from mazepy.geometry import Arena2D, GeometryError


def test_whole_segment_detects_thin_wall(detour_arena):
    assert not detour_arena.segment_is_valid((1, 1), (9, 1))
    assert detour_arena.segment_is_valid((1, 9), (9, 9))


def test_concave_outer_boundary_checks_complete_segment():
    arena = Arena2D([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    assert not arena.segment_is_valid((3, 0.5), (0.5, 3))


def test_boundary_following_is_explicit(open_arena):
    assert open_arena.segment_is_valid((0, 1), (0, 9))
    assert not open_arena.segment_is_valid((0, 1), (0, 9), allow_touch=False)
    assert not open_arena.segment_is_valid((-1, 1), (2, 1))


def test_dangling_tip_tangency(detour_arena):
    assert detour_arena.segment_is_valid((4, 8), (6, 8))
    assert not detour_arena.segment_is_valid((4, 8), (6, 8), allow_touch=False)


def test_full_wall_cannot_be_bypassed_at_boundary(boundary):
    arena = Arena2D(boundary, [(5, 0, 5, 10)])
    assert not arena.segment_is_valid((4, 0), (6, 0))
    assert arena.segment_is_valid((4, 1), (5, 0))
    assert arena.segment_is_valid((5, 0), (6, 1))
    assert not arena.path_is_valid([(4, 1), (5, 0), (6, 1)])
    assert not arena.path_is_valid([(4, 1), (5, 0), (5, 0), (6, 1)])


def test_t_junction_has_no_point_portal(boundary):
    arena = Arena2D(boundary, [(0, 5, 10, 5), (5, 0, 5, 5)])
    assert not arena.segment_is_valid((2, 2), (8, 8))
    assert not arena.path_is_valid([(2, 2), (5, 5), (8, 2)])


def test_cross_junction_has_no_point_portal(boundary):
    arena = Arena2D(boundary, [(5, 2, 5, 8), (2, 5, 8, 5)])
    assert not arena.segment_is_valid((4, 4), (6, 6))
    assert not arena.path_is_valid([(4, 4), (5, 5), (6, 6)])


def test_l_corner_uses_correct_free_sector(boundary):
    arena = Arena2D(boundary, [(2, 2, 8, 2), (2, 2, 2, 8)])
    assert arena.segment_is_valid((1, 3), (3, 1))
    assert not arena.segment_is_valid((3, 3), (1, 1))
    assert not arena.path_is_valid([(3, 3), (2, 2), (1, 1)])


def test_cannot_switch_sides_along_a_continuous_wall(boundary):
    arena = Arena2D(
        boundary,
        [(5, 0, 5, 10), (5, 3, 8, 3), (2, 7, 5, 7)],
    )
    assert not arena.segment_is_valid((5, 1), (5, 9))


def test_opposite_isolated_tangencies_can_use_open_space(boundary):
    arena = Arena2D(boundary, [(3, 5, 3, 8), (7, 2, 7, 5)])
    assert arena.segment_is_valid((1, 5), (9, 5))


def test_slanted_wall_crossing_and_tip(boundary):
    arena = Arena2D(boundary, [(2, 2, 8, 8)])
    assert not arena.segment_is_valid((3, 7), (7, 3))
    assert arena.segment_is_valid((1, 3), (3, 1))


def test_degenerate_and_malformed_paths(open_arena):
    assert open_arena.path_is_valid([[1, 1]])
    assert open_arena.path_is_valid([[1, 1], [1, 1]])
    assert open_arena.segment_is_valid((1, 1), (1, 1))
    assert not open_arena.segment_is_valid((-1, 1), (-1, 1))
    with pytest.raises(GeometryError):
        open_arena.path_is_valid([])
    with pytest.raises(GeometryError):
        open_arena.path_is_valid([1, 1])
    with pytest.raises(GeometryError):
        open_arena.segment_is_valid((np.nan, 1), (1, 1))
