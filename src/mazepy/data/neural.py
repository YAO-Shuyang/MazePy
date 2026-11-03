"""NumPy-backed signal containers, independent of analysis, IO, and Qt.

Values use (signal, sample) orientation. Event timestamps stay sparse. All time
values are seconds in the named recording clock; no synchronization is inferred.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import freeze, ids, positions, readonly, real_array, signal_id, window
from .time import TimeAxis, TimeIntervals

__all__ = [
    "NeuralRecording",
    "SampledActivity",
    "CalciumTrace",
    "SpikeEvents",
    "BinnedActivity",
]

_KINDS = {
    "fluorescence",
    "corrected_fluorescence",
    "dff",
    "calcium",
    "voltage",
    "optical_voltage",
    "current",
    "count",
    "binary",
    "event_weight",
    "expected_count",
    "rate",
    "probability",
    "phase",
    "amplitude",
    "power",
    "spike_events",
}
_NONNEGATIVE = {"count", "binary", "event_weight", "expected_count", "rate", "power"}


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class NeuralRecording(ABC):
    """Shared identity, units, provenance, and selection interface.

    ``signal_ids`` label rows, independently of source ROI/cluster numbering.
    ``signal_metadata`` maps column names to one entry per signal. Generic
    metadata accepts JSON-like values and numerical arrays and is frozen deeply.
    Observation intervals describe declared recording support, not sample dwell
    times. ``None`` means unknown and must not be interpreted as zero duration.
    """

    unit: str
    signal_kind: str
    signal_ids: Sequence[str | int] | None = None
    clock: str = "recording"
    observation_intervals: TimeIntervals | ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    signal_metadata: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def _initialize(self, n_signals):
        for name in ("unit", "clock"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.signal_kind not in _KINDS:
            raise ValueError(f"Unknown signal_kind: {self.signal_kind!r}.")
        labels = ids(range(n_signals) if self.signal_ids is None else self.signal_ids)
        if len(labels) != n_signals:
            raise ValueError("signal_ids must match the number of signals.")
        object.__setattr__(self, "signal_ids", labels)
        intervals = self.observation_intervals
        if intervals is not None and not isinstance(intervals, TimeIntervals):
            intervals = TimeIntervals(intervals)
        object.__setattr__(self, "observation_intervals", intervals)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", freeze(self.metadata))
        if not isinstance(self.signal_metadata, Mapping):
            raise TypeError("signal_metadata must be a mapping of aligned columns.")
        columns = {}
        for name, column in self.signal_metadata.items():
            if not isinstance(name, str):
                raise TypeError("Signal metadata column names must be strings.")
            if isinstance(column, str) or len(column) != n_signals:
                raise ValueError("Signal metadata columns must match the signal axis.")
            columns[name] = tuple(freeze(value) for value in column)
        object.__setattr__(self, "signal_metadata", freeze(columns))

    @property
    def n_signals(self) -> int:
        return len(self.signal_ids)

    def select_signals(self, selection) -> NeuralRecording:
        """Select/reorder by signal ID, keeping metadata and empty signals aligned."""
        if isinstance(selection, (str, int, np.integer)):
            selection = [selection]
        requested = ids(selection)
        lookup = {label: index for index, label in enumerate(self.signal_ids)}
        try:
            indices = np.asarray([lookup[label] for label in requested], dtype=np.intp)
        except KeyError as exc:
            raise KeyError(f"Unknown signal ID: {exc.args[0]!r}.") from None
        columns = {
            name: tuple(column[index] for index in indices)
            for name, column in self.signal_metadata.items()
        }
        return self._select_signals(indices, requested, columns)

    def with_metadata(self, **updates) -> NeuralRecording:
        """Return a new recording with updated provenance; preserve all data."""
        return replace(self, metadata={**self.metadata, **updates})

    @abstractmethod
    def _select_signals(self, indices, requested, columns):
        raise NotImplementedError

    @abstractmethod
    def time_slice(self, start: float, stop: float) -> NeuralRecording:
        """Return the part of the recording in a half-open time window."""
        raise NotImplementedError


def _matrix(recording, values, valid, *, time=None, additional_valid=None):
    array = real_array(values, "values", ndim=2)
    recording._initialize(array.shape[0])
    finite = np.isfinite(array)
    if valid is None:
        mask = finite
    else:
        valid = np.asarray(valid)
        if valid.dtype.kind != "b":
            raise TypeError("valid must be a boolean mask.")
        if valid.shape == (array.shape[1],):
            valid = np.broadcast_to(valid, array.shape)
        if valid.shape != array.shape:
            raise ValueError("valid must have shape (samples,) or (signals, samples).")
        mask = valid & finite
    if time is not None and recording.observation_intervals is not None:
        mask &= recording.observation_intervals.contains(time.values)[None, :]
    if additional_valid is not None:
        mask &= additional_valid
    kind = recording.signal_kind
    if kind == "spike_events":
        raise ValueError("Sampled matrices cannot have kind 'spike_events'.")
    if kind == "count" and array.dtype.kind not in "iu":
        raise TypeError("Counts must use an integer dtype; preserve inferred floats.")
    if kind in _NONNEGATIVE and np.any(array < 0, where=mask):
        raise ValueError(f"Valid {kind} values must be non-negative.")
    if kind == "binary" and np.any((array != 0) & (array != 1), where=mask):
        raise ValueError("Binary activity must contain only zero or one.")
    if kind == "probability" and np.any((array < 0) | (array > 1), where=mask):
        raise ValueError("Probabilities must lie in [0, 1].")
    object.__setattr__(recording, "values", readonly(array))
    object.__setattr__(recording, "valid", readonly(mask))


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SampledActivity(NeuralRecording):
    """A dense N-by-T signal matrix, timestamps, and per-signal validity.

    Numeric dtype and missing values are preserved. Non-finite values are marked
    invalid independently for each signal. Construction owns immutable data;
    use ``to_numpy()`` for a writable copy. This initial container is eager,
    rather than a lazy raw-acquisition or image-movie reader.
    """

    values: ArrayLike
    time: TimeAxis | ArrayLike
    valid: ArrayLike | None = None

    def __post_init__(self):
        axis = self.time if isinstance(self.time, TimeAxis) else TimeAxis(self.time)
        array = real_array(self.values, "values", ndim=2)
        if array.shape[1] != axis.n_samples:
            raise ValueError("The time axis must match values.shape[1].")
        object.__setattr__(self, "time", axis)
        _matrix(self, array, self.valid, time=axis)

    @property
    def n_samples(self) -> int:
        return self.time.n_samples

    def _select_signals(self, indices, requested, columns):
        return replace(
            self,
            values=self.values[indices],
            valid=self.valid[indices],
            signal_ids=requested,
            signal_metadata=columns,
        )

    def select_samples(self, selection) -> SampledActivity:
        """Select ordered samples, without filling gaps or inventing exposure.

        Declared recording intervals are retained as source context. Selecting
        samples does not imply continuous valid coverage between them.
        """
        indices = positions(selection, self.n_samples)
        return replace(
            self,
            values=self.values[:, indices],
            valid=self.valid[:, indices],
            time=self.time.select_samples(indices),
        )

    def time_slice(self, start: float, stop: float) -> SampledActivity:
        start, stop = window(start, stop)
        selected = self.select_samples(self.time.indices_between(start, stop))
        intervals = self.observation_intervals
        if intervals is not None:
            selected = replace(
                selected, observation_intervals=intervals.clip(start, stop)
            )
        return selected

    def to_numpy(self, *, copy: bool = True) -> NDArray:
        """Return a writable copy, or the immutable stored array when copy=False."""
        return np.array(self.values, copy=True) if copy else self.values


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class CalciumTrace(SampledActivity):
    """Calcium fluorescence/dF/F/calcium estimates; inference remains explicit.

    ``signal_kind`` can also identify externally inferred event weights, expected
    counts, probabilities, or rates. No baseline, deconvolution, or threshold is
    applied implicitly, and no inferred amplitude is rounded to an integer.
    """

    signal_kind: str = "fluorescence"

    def __post_init__(self):
        allowed = {
            "fluorescence",
            "corrected_fluorescence",
            "dff",
            "calcium",
            "event_weight",
            "expected_count",
            "rate",
            "probability",
            "binary",
        }
        if self.signal_kind not in allowed:
            raise ValueError(
                "CalciumTrace requires an explicit calcium representation."
            )
        SampledActivity.__post_init__(self)


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SpikeEvents(NeuralRecording):
    """Sparse spike times and zero-based indices into a stable signal-ID roster.

    Declare observation intervals explicitly, including silent time. Times are
    sorted non-decreasingly; coincident spikes are preserved. A roster may
    include units with no spikes. Use ``from_unit_ids`` for source cluster labels.
    """

    times: ArrayLike
    unit_indices: ArrayLike
    unit: str = "spikes"
    signal_kind: str = "spike_events"
    unit_observation_intervals: Mapping[str | int, TimeIntervals | ArrayLike] = field(
        default_factory=dict
    )

    def __post_init__(self):
        times = np.asarray(
            real_array(self.times, "spike times", ndim=1), dtype=np.float64
        )
        indices = real_array(self.unit_indices, "unit_indices", ndim=1)
        if indices.dtype.kind not in "iu":
            if indices.size:
                raise TypeError("unit_indices must be integers.")
            indices = np.empty(0, dtype=np.int64)
        if times.size != indices.size:
            raise ValueError("Each spike must have one unit index.")
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("Spike times must be finite and non-decreasing.")
        if self.signal_ids is None:
            unique = np.unique(indices)
            if not np.array_equal(unique, np.arange(unique.size)):
                raise ValueError(
                    "Non-contiguous indices need an explicit signal-ID roster; "
                    "use from_unit_ids for source cluster labels."
                )
            n = unique.size
        else:
            n = len(self.signal_ids)
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("unit_indices must refer to the signal-ID roster.")
        self._initialize(n)
        if self.signal_kind != "spike_events" or self.unit != "spikes":
            raise ValueError("SpikeEvents stores spike events with unit='spikes'.")
        if self.observation_intervals is None:
            raise ValueError(
                "Declare observation_intervals; do not use the last spike."
            )
        if not np.all(self.observation_intervals.contains(times)):
            raise ValueError("Spikes fall outside the declared observation intervals.")
        if not isinstance(self.unit_observation_intervals, Mapping):
            raise TypeError("unit_observation_intervals must be a mapping.")
        per_unit = {}
        for label, support in self.unit_observation_intervals.items():
            label = signal_id(label)
            if label not in self.signal_ids:
                raise ValueError("Per-unit intervals reference an unknown unit.")
            support = (
                support
                if isinstance(support, TimeIntervals)
                else TimeIntervals(support)
            )
            for left, right in support.values:
                overlap = self.observation_intervals.clip(left, right).duration
                if not np.isclose(overlap, right - left, rtol=0, atol=1e-12):
                    raise ValueError(
                        "Per-unit support exceeds recording observation time."
                    )
            unit_index = self.signal_ids.index(label)
            if not np.all(support.contains(times[indices == unit_index])):
                raise ValueError(
                    "Spikes fall outside their unit's observation intervals."
                )
            per_unit[label] = support
        object.__setattr__(
            self, "unit_observation_intervals", MappingProxyType(per_unit)
        )
        object.__setattr__(self, "times", readonly(times, dtype=np.float64))
        object.__setattr__(self, "unit_indices", readonly(indices, dtype=np.int64))

    @classmethod
    def from_unit_ids(
        cls,
        *,
        times: ArrayLike,
        unit_ids: Sequence[str | int],
        signal_ids: Sequence[str | int] | None = None,
        **kwargs,
    ) -> SpikeEvents:
        """Map source unit labels without assuming contiguous or sorted IDs."""
        events = tuple(signal_id(value) for value in unit_ids)
        roster = ids(dict.fromkeys(events) if signal_ids is None else signal_ids)
        lookup = {label: index for index, label in enumerate(roster)}
        try:
            indices = np.asarray([lookup[label] for label in events], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Unknown spike unit ID: {exc.args[0]!r}.") from None
        return cls(times=times, unit_indices=indices, signal_ids=roster, **kwargs)

    @property
    def n_events(self) -> int:
        return len(self.times)

    def _select_signals(self, indices, requested, columns):
        lookup = np.full(self.n_signals, -1, dtype=np.int64)
        lookup[indices] = np.arange(len(indices))
        remapped = lookup[self.unit_indices]
        keep = remapped >= 0
        return replace(
            self,
            times=self.times[keep],
            unit_indices=remapped[keep],
            signal_ids=requested,
            signal_metadata=columns,
            unit_observation_intervals={
                label: self.unit_observation_intervals[label]
                for label in requested
                if label in self.unit_observation_intervals
            },
        )

    def time_slice(self, start: float, stop: float) -> SpikeEvents:
        start, stop = window(start, stop)
        left, right = np.searchsorted(self.times, [start, stop], side="left")
        return replace(
            self,
            times=self.times[left:right],
            unit_indices=self.unit_indices[left:right],
            observation_intervals=self.observation_intervals.clip(start, stop),
            unit_observation_intervals={
                label: intervals.clip(start, stop)
                for label, intervals in self.unit_observation_intervals.items()
            },
        )


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class BinnedActivity(NeuralRecording):
    """N-by-K activity with explicit, non-overlapping time-bin edges.

    Exposure is optional known observation time in seconds, either K or N-by-K.
    Unknown exposure remains None. Partial bins cannot be sliced as though their
    counts/rates were observations of a shorter interval; rebin the source.
    """

    values: ArrayLike
    bin_edges: ArrayLike
    valid: ArrayLike | None = None
    exposure: ArrayLike | None = None
    signal_kind: str = "count"
    unit: str | None = None

    def __post_init__(self):
        if self.unit is None:
            if self.signal_kind != "count":
                raise ValueError(
                    "Specify units explicitly for non-count binned activity."
                )
            object.__setattr__(self, "unit", "spikes")
        array = real_array(self.values, "values", ndim=2)
        edges = real_array(self.bin_edges, "bin_edges", ndim=1)
        edges = np.asarray(edges, dtype=np.float64)
        if len(edges) != array.shape[1] + 1 or not np.isfinite(edges).all():
            raise ValueError("Bin edges must be finite with length n_bins + 1.")
        with np.errstate(over="ignore"):
            widths = np.diff(edges)
        if not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("Bin edges must increase with finite bin durations.")
        object.__setattr__(self, "bin_edges", readonly(edges))
        self._initialize(array.shape[0])
        available = widths.copy()
        if self.observation_intervals is not None:
            available = np.zeros(array.shape[1], dtype=np.float64)
            for start, stop in self.observation_intervals.values:
                available += np.maximum(
                    0, np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start)
                )
        support_mask = np.broadcast_to(available > 0, array.shape).copy()
        if self.exposure is not None:
            exposure = real_array(
                self.exposure, "exposure", ndim=np.ndim(self.exposure)
            )
            if exposure.shape == (array.shape[1],):
                exposure = np.broadcast_to(exposure, array.shape)
            if exposure.shape != array.shape:
                raise ValueError("Exposure must have shape (bins,) or (signals, bins).")
            if not np.isfinite(exposure).all() or np.any(exposure < 0):
                raise ValueError("Exposure must be finite and non-negative.")
            if np.any(exposure > available[None, :] + 1e-12):
                raise ValueError("Exposure exceeds the declared observation support.")
            object.__setattr__(self, "exposure", readonly(exposure, dtype=np.float64))
            support_mask &= exposure > 0
        _matrix(self, array, self.valid, additional_valid=support_mask)

    @property
    def n_bins(self) -> int:
        return self.values.shape[1]

    @property
    def time(self) -> TimeAxis:
        """Bin-center timestamps, distinct from bin edges and exposure."""
        return TimeAxis(self.bin_edges[:-1] + np.diff(self.bin_edges) / 2)

    def _select_signals(self, indices, requested, columns):
        return replace(
            self,
            values=self.values[indices],
            valid=self.valid[indices],
            signal_ids=requested,
            signal_metadata=columns,
            exposure=None if self.exposure is None else self.exposure[indices],
        )

    def select_bins(self, selection) -> BinnedActivity:
        """Select a contiguous bin range without changing bin meanings."""
        indices = positions(selection, self.n_bins)
        if indices.size and np.any(np.diff(indices) != 1):
            raise ValueError("Binned activity requires a contiguous bin selection.")
        if indices.size:
            start = int(indices[0])
        elif isinstance(selection, slice):
            start = min(self.n_bins, max(0, selection.indices(self.n_bins)[0]))
        else:
            start = 0
        edges = self.bin_edges[start : start + indices.size + 1]
        intervals = self.observation_intervals
        if intervals is not None:
            intervals = intervals.clip(float(edges[0]), float(edges[-1]))
        return replace(
            self,
            values=self.values[:, indices],
            valid=self.valid[:, indices],
            bin_edges=edges,
            observation_intervals=intervals,
            exposure=None if self.exposure is None else self.exposure[:, indices],
        )

    def time_slice(self, start: float, stop: float) -> BinnedActivity:
        start, stop = window(start, stop)
        low, high = float(self.bin_edges[0]), float(self.bin_edges[-1])
        start, stop = np.clip([start, stop], low, high)
        selected = []
        for bound in (start, stop):
            index = int(np.searchsorted(self.bin_edges, bound))
            candidates = [i for i in (index - 1, index) if 0 <= i < len(self.bin_edges)]
            match = next(
                (
                    i
                    for i in candidates
                    if np.isclose(self.bin_edges[i], bound, rtol=0, atol=1e-12)
                ),
                None,
            )
            if match is None:
                raise ValueError("Time slice cuts a bin; rebin original observations.")
            selected.append(match)
        left, right = selected
        if left == right:
            return replace(
                self,
                values=self.values[:, :0],
                valid=self.valid[:, :0],
                bin_edges=self.bin_edges[left : left + 1],
                exposure=None if self.exposure is None else self.exposure[:, :0],
                observation_intervals=None
                if self.observation_intervals is None
                else self.observation_intervals.clip(float(start), float(stop)),
            )
        return self.select_bins(slice(left, right))

    def to_numpy(self, *, copy: bool = True) -> NDArray:
        return np.array(self.values, copy=True) if copy else self.values
