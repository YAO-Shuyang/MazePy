from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mazepy.geometry import Arena2D, GeometryError


def test_empty_walls_are_a_stable_array(open_arena):
    assert open_arena.walls.shape == (0, 4)
    assert open_arena.area == 100
    assert open_arena.bounds == (0, 0, 10, 10)


def test_scalar_and_batch_containment(detour_arena):
    assert detour_arena.contains((1, 1)) is True
    assert detour_arena.contains((5, 4)) is False
    assert detour_arena.contains((5, 4), include_boundary=True) is True
    points = [(1, 1), (11, 1), (0, 2), (np.nan, 1), (1, np.inf)]
    np.testing.assert_array_equal(
        detour_arena.contains(points), [True, False, False, False, False]
    )
    assert detour_arena.contains(np.empty((0, 2))).shape == (0,)


def test_input_and_returned_coordinates_cannot_mutate_geometry(boundary):
    source = np.array(boundary, dtype=float)
    arena = Arena2D(source)
    source[:] = 0
    assert arena.area == 100
    with pytest.raises(ValueError):
        arena.boundary[0, 0] = 1
    with pytest.raises(ValueError):
        arena.boundary.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        arena.unit = "m"


def test_closed_and_clockwise_rings_have_same_fingerprint(boundary):
    a = Arena2D(boundary)
    b = Arena2D([*boundary[::-1], boundary[-1]])
    assert a.fingerprint == b.fingerprint


def test_wall_order_and_direction_do_not_change_fingerprint(boundary):
    a = Arena2D(boundary, [(2, 0, 2, 7), (4, 3, 8, 3)])
    b = Arena2D(boundary, [(8, 3, 4, 3), (2, 7, 2, 0)])
    assert a.fingerprint == b.fingerprint


def test_edits_return_new_arenas(open_arena):
    blocked = open_arena.with_wall((5, 0, 5, 10))
    assert blocked.fingerprint != open_arena.fingerprint
    assert open_arena.distance((4, 1), (6, 1)) == 2
    assert np.isinf(blocked.distance((4, 1), (6, 1)))
    restored = blocked.without_wall(0)
    assert restored.fingerprint == open_arena.fingerprint
    with pytest.raises(IndexError):
        blocked.without_wall(-1)
    with pytest.raises(GeometryError):
        blocked.without_wall(True)


@pytest.mark.parametrize(
    "boundary",
    [
        [(0, 0), (1, 1)],
        [(0, 0), (1, 1), (0, 1), (1, 0)],
        [(0, 0), (0, 0), (1, 1)],
        [(0, 0), (1, 0), (np.nan, 1)],
        [(0, 0), (1, 0), (2, 0)],
        np.zeros((3, 3)),
    ],
)
def test_bad_boundary_is_rejected(boundary):
    with pytest.raises(GeometryError):
        Arena2D(boundary)


@pytest.mark.parametrize(
    "walls",
    [[(1, 1, 1, 1)], [(1, 1, 11, 1)], [(1, np.inf, 2, 2)], [(1, 2, 3)]],
)
def test_bad_walls_are_rejected(boundary, walls):
    with pytest.raises(GeometryError):
        Arena2D(boundary, walls)


@pytest.mark.parametrize("tolerance", [0, -1, np.inf, np.nan])
def test_invalid_tolerance(boundary, tolerance):
    with pytest.raises(GeometryError):
        Arena2D(boundary, tolerance=tolerance)


def test_wall_shape_variants(boundary):
    a = Arena2D(boundary, [[[5, 0], [5, 8]]])
    assert a.walls.shape == (1, 4)
    with pytest.raises(GeometryError):
        a.with_wall([1, 2, 3])


def test_concave_boundary_checks_complete_wall():
    boundary = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    with pytest.raises(GeometryError):
        Arena2D(boundary, [(3, 0.5, 0.5, 3)])


def test_obstacles_are_solid_and_may_overlap(boundary):
    a = Arena2D(
        boundary,
        obstacles=[
            [(2, 2), (5, 2), (5, 5), (2, 5)],
            [(4, 4), (6, 4), (6, 6), (4, 6)],
        ],
    )
    assert a.area == 88
    assert not a.contains((3, 3))
    assert not a.contains((5.5, 5.5))
    with pytest.raises(GeometryError):
        Arena2D(boundary, obstacles=[[(-1, 1), (1, 1), (1, 2), (-1, 2)]])
    with pytest.raises(GeometryError):
        Arena2D(boundary, obstacles=[boundary])
    with pytest.raises(GeometryError):
        Arena2D(
            boundary,
            walls=[(1, 3, 6, 3)],
            obstacles=[[(2, 2), (4, 2), (4, 4), (2, 4)]],
        )


def test_units_are_explicit_labels(boundary):
    a = Arena2D(boundary, unit="m")
    assert a.unit == "m"
    assert a.distance((1, 1), (2, 1)) == 1
    assert a.fingerprint != Arena2D(boundary, unit="cm").fingerprint
    with pytest.raises(GeometryError):
        Arena2D(boundary, unit="")
