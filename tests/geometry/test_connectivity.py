"""Independent discrete-connectivity checks for square-bin wall mazes."""

import math
from collections import deque

import numpy as np
import pytest

from mazepy.geometry import Arena2D


@pytest.mark.parametrize("seed", range(8))
def test_continuous_reachability_matches_independent_flood_fill(seed):
    generator = np.random.default_rng(seed)
    size = 4
    neighbors = {i: [] for i in range(size**2)}
    walls = []
    for y in range(size):
        for x in range(size):
            node = y * size + x
            for dx, dy in [(1, 0), (0, 1)]:
                xx, yy = x + dx, y + dy
                if xx >= size or yy >= size:
                    continue
                other = yy * size + xx
                if generator.random() < 0.55:
                    if dx:
                        walls.append((x + 1, y, x + 1, y + 1))
                    else:
                        walls.append((x, y + 1, x + 1, y + 1))
                else:
                    neighbors[node].append(other)
                    neighbors[other].append(node)
    arena = Arena2D([(0, 0), (size, 0), (size, size), (0, size)], walls)
    for start in generator.choice(size**2, size=4, replace=False):
        reached = {start}
        queue = deque([start])
        while queue:
            for other in neighbors[queue.popleft()]:
                if other not in reached:
                    reached.add(other)
                    queue.append(other)
        for end in generator.choice(size**2, size=5, replace=False):
            p = (start % size + 0.5, start // size + 0.5)
            q = (end % size + 0.5, end // size + 0.5)
            distance = arena.distance(p, q)
            assert math.isfinite(distance) == (end in reached)
            if math.isfinite(distance):
                assert arena.path_is_valid(arena.shortest_path(p, q).points)


def test_polygonal_barrier_can_split_the_arena():
    arena = Arena2D(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        obstacles=[[(4, 0), (6, 0), (6, 10), (4, 10)]],
    )
    assert math.isinf(arena.distance((2, 5), (8, 5)))


def test_touching_solid_obstacles_do_not_create_a_point_portal():
    arena = Arena2D(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        obstacles=[
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            [(5, 5), (10, 5), (10, 10), (5, 10)],
        ],
    )
    assert not arena.segment_is_valid((2, 8), (8, 2))
    assert math.isinf(arena.distance((2, 8), (8, 2)))


@pytest.mark.parametrize("angle", [0.37, 1.8])
def test_rotated_solid_obstacle_and_sampled_boundary_following(angle):
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )

    def transform(points):
        return np.asarray(points) @ rotation.T + [13, -21]

    arena = Arena2D(
        transform([(0, 0), (10, 0), (10, 10), (0, 10)]),
        obstacles=[transform([(4, 4), (6, 4), (6, 6), (4, 6)])],
    )
    p, q = transform([(2, 5), (8, 5)])
    path = arena.shortest_path(p, q)
    assert path.length == pytest.approx(2 * math.sqrt(5) + 2)
    assert arena.path_is_valid(path.points)
    assert arena.path_is_valid(path.sample(0.1))
