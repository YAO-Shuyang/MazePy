"""Explicit sample timestamps and half-open observation intervals, in seconds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positions, readonly, real_array, window

__all__ = ["TimeAxis", "TimeIntervals"]


@dataclass(frozen=True, slots=True, eq=False)
class TimeIntervals:
    """Sorted, non-overlapping observation intervals ``[start, stop)`` in seconds.

    Adjacent intervals remain separate, preserving recording segment boundaries.
    Empty intervals are represented by an array of shape ``(0, 2)``.
    """

    values: ArrayLike

    def __post_init__(self):
        array = np.asarray(self.values)
        if array.shape == (0,):
            array = np.empty((0, 2))
        array = real_array(array, "intervals", ndim=2)
        if array.shape[1] != 2:
            raise ValueError("Intervals must have shape (n_intervals, 2).")
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array[:, 0] >= array[:, 1]):
            raise ValueError("Intervals must be finite with start < stop.")
        with np.errstate(over="ignore"):
            widths = array[:, 1] - array[:, 0]
        if not np.isfinite(widths).all():
            raise ValueError("Interval durations must be finite.")
        if np.any(array[1:, 0] < array[:-1, 1]):
            raise ValueError("Intervals must be sorted and non-overlapping.")
        object.__setattr__(self, "values", readonly(array))

    @property
    def duration(self) -> float:
        """Total declared observation time, excluding gaps."""
        return float(np.sum(self.values[:, 1] - self.values[:, 0]))

    def contains(self, times: ArrayLike) -> bool | NDArray[np.bool_]:
        """Return membership, with each right boundary excluded."""
        times = np.asarray(times)
        if times.dtype.kind not in "biuf":
            raise TypeError("Membership queries must contain real numerical times.")
        times = np.asarray(times, dtype=np.float64)
        if not len(self.values):
            result = np.zeros(times.shape, dtype=bool)
        else:
            indices = np.searchsorted(self.values[:, 0], times, side="right") - 1
            right = self.values[np.clip(indices, 0, len(self.values) - 1), 1]
            result = (indices >= 0) & (times < right) & np.isfinite(times)
        return bool(result) if result.ndim == 0 else result

    def clip(self, start: float, stop: float) -> TimeIntervals:
        """Intersect observation intervals with a half-open time window."""
        start, stop = window(start, stop)
        left = np.maximum(self.values[:, 0], start)
        right = np.minimum(self.values[:, 1], stop)
        keep = left < right
        return TimeIntervals(np.column_stack((left[keep], right[keep])))


@dataclass(frozen=True, slots=True, eq=False)
class TimeAxis:
    """A finite, strictly increasing sample-time vector in seconds.

    Timestamps do not by themselves define sample exposure or recording duration.
    No timestamps are sorted, removed, or extended automatically.
    """

    values: ArrayLike

    def __post_init__(self):
        array = real_array(self.values, "timestamps", ndim=1)
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all() or np.any(np.diff(array) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing.")
        object.__setattr__(self, "values", readonly(array))

    @classmethod
    def regular(
        cls, n_samples: int, sampling_rate: float, *, start: float = 0.0
    ) -> TimeAxis:
        """Construct explicit timestamps at a declared sampling rate in Hz."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be an integer.")
        if n_samples < 0 or isinstance(sampling_rate, bool):
            raise ValueError("Sample count must be non-negative and rate positive.")
        sampling_rate, start = float(sampling_rate), float(start)
        if not np.isfinite([sampling_rate, start]).all() or sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive and start finite.")
        return cls(start + np.arange(n_samples, dtype=np.float64) / sampling_rate)

    @property
    def n_samples(self) -> int:
        return len(self.values)

    def select_samples(self, selection) -> TimeAxis:
        """Select samples in increasing order, preserving their original times."""
        return TimeAxis(self.values[positions(selection, self.n_samples)])

    def indices_between(self, start: float, stop: float) -> slice:
        """Locate sample indices within ``[start, stop)``."""
        start, stop = window(start, stop)
        return slice(
            int(np.searchsorted(self.values, start, side="left")),
            int(np.searchsorted(self.values, stop, side="left")),
        )
