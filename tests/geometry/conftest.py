import pytest

from mazepy.geometry import Arena2D


@pytest.fixture
def boundary():
    return [(0, 0), (10, 0), (10, 10), (0, 10)]


@pytest.fixture
def open_arena(boundary):
    return Arena2D(boundary)


@pytest.fixture
def detour_arena(boundary):
    return Arena2D(boundary, walls=[(5, 0, 5, 8)])
