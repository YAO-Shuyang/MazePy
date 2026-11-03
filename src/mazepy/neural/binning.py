"""Half-open spike counting and exposure-normalized firing rates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mazepy.data import BinnedActivity, SpikeEvents, TimeIntervals

__all__ = ["bin_spikes", "to_rate"]


def _edges(values: ArrayLike) -> NDArray[np.float64]:
    edges = np.asarray(values)
    if edges.ndim != 1 or edges.size == 0:
        raise ValueError("bin_edges must be a non-empty one-dimensional vector.")
    if edges.dtype.kind not in "iuf":
        raise TypeError("bin_edges must contain real numerical seconds.")
    edges = np.asarray(edges, dtype=np.float64)
    with np.errstate(over="ignore"):
        widths = np.diff(edges)
    if (
        not np.isfinite(edges).all()
        or not np.isfinite(widths).all()
        or np.any(widths <= 0)
    ):
        raise ValueError("bin_edges must increase with finite positive durations.")
    return edges


def _exposure(support: TimeIntervals, edges: NDArray) -> NDArray[np.float64]:
    """Sum direct overlaps, without a bins-by-intervals matrix or time subtraction.

    Only bins intersecting each sorted interval are visited. Direct overlap
    avoids subtracting large accumulated durations to recover a tiny bin.
    """
    result = np.zeros(len(edges) - 1, dtype=np.float64)
    for start, stop in support.values:
        first = max(0, int(np.searchsorted(edges, start, side="right")) - 1)
        last = min(result.size, int(np.searchsorted(edges, stop, side="left")))
        if first < last:
            result[first:last] += np.minimum(edges[first + 1 : last + 1], stop) - (
                np.maximum(edges[first:last], start)
            )
    return result


def bin_spikes(spikes: SpikeEvents, bin_edges: ArrayLike) -> BinnedActivity:
    """Count spikes in explicit bins ``[left, right)`` in the recording clock.

    The final right edge is excluded, including for the final bin. Events outside
    the requested window are ignored. Counts are int64, retain coincident spikes,
    and preserve the complete unit roster, including silent units.

    Exposure is each bin's overlap with declared observation intervals, using
    per-unit support where supplied. Zero-exposure cells contain raw count zero
    and are invalid; a valid zero count represents observed silence. Partial
    exposure remains valid and is retained in seconds. A single edge produces
    zero bins. IDs, clock, and source metadata are preserved.
    """
    if not isinstance(spikes, SpikeEvents):
        raise TypeError("spikes must be a SpikeEvents recording.")
    edges = _edges(bin_edges)
    n_bins = len(edges) - 1
    left, right = np.searchsorted(spikes.times, [edges[0], edges[-1]], side="left")
    times = spikes.times[left:right]
    bins = np.searchsorted(edges, times, side="right") - 1
    flat = spikes.unit_indices[left:right] * n_bins + bins
    counts = np.bincount(flat, minlength=spikes.n_signals * n_bins).reshape(
        spikes.n_signals, n_bins
    )
    support = spikes.observation_intervals.clip(float(edges[0]), float(edges[-1]))
    exposure = np.broadcast_to(_exposure(support, edges), counts.shape).copy()
    lookup = {label: index for index, label in enumerate(spikes.signal_ids)}
    for label, intervals in spikes.unit_observation_intervals.items():
        exposure[lookup[label]] = _exposure(intervals, edges)
    return BinnedActivity(
        values=counts.astype(np.int64, copy=False),
        bin_edges=edges,
        exposure=exposure,
        signal_ids=spikes.signal_ids,
        observation_intervals=support,
        clock=spikes.clock,
        metadata=spikes.metadata,
        signal_metadata=spikes.signal_metadata,
    )


def to_rate(counts: BinnedActivity) -> BinnedActivity:
    """Convert integer spike counts to unsmoothed firing rates in Hz.

    Requires ``signal_kind='count'``, ``unit='spikes'``, and explicit exposure.
    Divide each valid cell by its observation time, not by nominal bin width.
    Invalid or unobserved cells become NaN and remain invalid. No smoothing,
    interpolation, calcium conversion, or exposure inference is performed.
    """
    if not isinstance(counts, BinnedActivity):
        raise TypeError("counts must be BinnedActivity.")
    if counts.signal_kind != "count" or counts.unit != "spikes":
        raise ValueError("to_rate requires integer spike counts with unit='spikes'.")
    if counts.exposure is None:
        raise ValueError("Explicit observation exposure is required to estimate rates.")
    values = np.full(counts.values.shape, np.nan, dtype=np.float64)
    valid = counts.valid & (counts.exposure > 0)
    with np.errstate(over="ignore"):
        np.divide(counts.values, counts.exposure, out=values, where=valid)
    if np.any(~np.isfinite(values), where=valid):
        raise ValueError("Firing rate exceeds finite float64 range.")
    return BinnedActivity(
        values=values,
        bin_edges=counts.bin_edges,
        exposure=counts.exposure,
        valid=valid,
        signal_kind="rate",
        unit="Hz",
        signal_ids=counts.signal_ids,
        observation_intervals=counts.observation_intervals,
        clock=counts.clock,
        metadata=counts.metadata,
        signal_metadata=counts.signal_metadata,
    )
