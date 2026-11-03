import numpy as np
import pytest

from mazepy.data import TimeAxis, TimeIntervals


def test_intervals_preserve_gaps_adjacency_and_half_open_boundaries():
    support = TimeIntervals([[0, 1], [1, 2], [4, 5]])
    assert support.duration == 3
    np.testing.assert_array_equal(
        support.contains([-1, 0, 1, 2, 4, 5, np.nan, np.inf]),
        [False, True, True, False, True, False, False, False],
    )
    assert support.contains(0) is True
    assert support.contains(5) is False
    np.testing.assert_allclose(
        support.clip(0.5, 4.5).values, [[0.5, 1], [1, 2], [4, 4.5]]
    )
    assert support.clip(2, 4).values.shape == (0, 2)
    assert TimeIntervals([]).duration == 0
    assert TimeIntervals([]).contains(0) is False
    assert support.clip(0.2, 0.2).duration == 0


@pytest.mark.parametrize(
    "values",
    [
        [[1, 1]],
        [[2, 1]],
        [[0, 2], [1, 3]],
        [[4, 5], [0, 1]],
        [[0, np.inf]],
        [1, 2],
        [[0, 1, 2]],
    ],
)
def test_invalid_intervals(values):
    with pytest.raises(ValueError):
        TimeIntervals(values)


def test_regular_irregular_empty_and_singleton_time_axes():
    axis = TimeAxis.regular(4, 2, start=10)
    np.testing.assert_array_equal(axis.values, [10, 10.5, 11, 11.5])
    assert axis.n_samples == 4
    indices = axis.indices_between(10.5, 11.5)
    np.testing.assert_array_equal(axis.values[indices], [10.5, 11])
    np.testing.assert_array_equal(axis.select_samples([0, 3]).values, [10, 11.5])
    assert TimeAxis([]).n_samples == 0
    assert TimeAxis([10]).n_samples == 1
    assert TimeAxis.regular(0, 30).n_samples == 0


@pytest.mark.parametrize("values", [[1, 1], [2, 1], [0, np.nan], [[0, 1]]])
def test_invalid_time_axes(values):
    with pytest.raises(ValueError):
        TimeAxis(values)


@pytest.mark.parametrize("n,rate", [(-1, 10), (2, 0), (2, -1), (2, np.inf), (2, True)])
def test_invalid_regular_axes(n, rate):
    with pytest.raises(ValueError):
        TimeAxis.regular(n, rate)


def test_clock_windows_are_explicit():
    with pytest.raises(TypeError):
        TimeAxis.regular(1.5, 20)
    for start, stop in [(2, 1), (0, np.inf), (np.nan, 1)]:
        with pytest.raises(ValueError):
            TimeAxis([0, 1]).indices_between(start, stop)


def test_extreme_interval_duration_overflow_is_rejected():
    with pytest.raises(ValueError, match="durations must be finite"):
        TimeIntervals([[-1e308, 1e308]])
