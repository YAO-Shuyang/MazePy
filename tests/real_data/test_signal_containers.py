"""Invoke the public containers on published paired calcium/spike measurements.

References are direct SciPy source fields and independent NumPy masks/histograms.
This validates storage/selection contracts, not inference or pipeline readers.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from mazepy.data import (
    BinnedActivity,
    CalciumTrace,
    SampledActivity,
    SpikeEvents,
    TimeAxis,
    TimeIntervals,
)

pytestmark = pytest.mark.real_data
DEMO = Path(__file__).resolve().parents[2] / "examples/data/cascade_ds01"


@pytest.fixture(scope="module")
def native():
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    path = DEMO / manifest["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    source = loadmat(path, squeeze_me=True, struct_as_record=False)["CAttached"]
    record = np.asarray(source, dtype=object).reshape(-1)[0]
    times = np.asarray(record.fluo_time, dtype=np.float64).reshape(-1)
    calcium = np.asarray(record.fluo_mean, dtype=np.float64).reshape(-1)
    spikes = np.asarray(record.events_AP, dtype=np.float64).reshape(-1) / 10_000
    observed = spikes[(spikes >= times[0]) & (spikes < times[-1])]
    assert times.size == 1164
    assert spikes.size == 44
    assert observed.size == 43
    return times, calcium, observed


def test_time_axis_and_support_against_recorded_timestamps(native):
    times, _, _ = native
    axis = TimeAxis(times)
    assert axis.n_samples == 1164
    np.testing.assert_array_equal(axis.values, times)
    start, stop = float(times[100]), float(times[900])
    expected = times[(times >= start) & (times < stop)]
    np.testing.assert_array_equal(
        axis.values[axis.indices_between(start, stop)], expected
    )
    np.testing.assert_array_equal(
        axis.select_samples([0, 100, 900]).values, times[[0, 100, 900]]
    )
    support = TimeIntervals([[float(times[0]), float(times[-1])]])
    assert support.duration == pytest.approx(float(times[-1] - times[0]), abs=1e-12)
    np.testing.assert_array_equal(
        support.contains(times), (times >= times[0]) & (times < times[-1])
    )
    clipped = support.clip(start, stop)
    np.testing.assert_array_equal(clipped.values, [[start, stop]])
    assert clipped.duration == pytest.approx(stop - start, abs=1e-12)
    # A source-derived regular grid is a generator check, not the native clock.
    rate = 1 / float(np.median(np.diff(times)))
    regular = TimeAxis.regular(len(times), rate, start=float(times[0]))
    expected_grid = times[0] + np.arange(len(times)) / rate
    np.testing.assert_allclose(regular.values, expected_grid, rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", [SampledActivity, CalciumTrace])
def test_sampled_public_operations_preserve_recorded_signal(native, cls):
    times, calcium, _ = native
    activity = cls(
        values=calcium[None, :],
        time=times,
        signal_ids=["theis16-cell21"],
        unit="dF/F",
        signal_kind="dff",
        clock="calcium-native",
        metadata={"fixture": "cascade_ds01_theis16_cell21"},
        signal_metadata={"source_cell": [21]},
    )
    assert activity.n_signals == 1
    assert activity.n_samples == 1164
    np.testing.assert_array_equal(activity.values[0], calcium)
    assert activity.valid.all()
    assert activity.observation_intervals is None
    assert np.any(activity.values != activity.values.astype(int))
    selected = activity.select_signals("theis16-cell21")
    assert type(selected) is cls
    assert selected.signal_metadata["source_cell"] == (21,)
    np.testing.assert_array_equal(selected.to_numpy()[0], calcium)
    np.testing.assert_array_equal(activity.to_numpy(copy=False)[0], calcium)
    samples = np.arange(50, 300, 3)
    np.testing.assert_array_equal(
        activity.select_samples(samples).values[0], calcium[samples]
    )
    start, stop = times[100], times[900]
    reference = (times >= start) & (times < stop)
    sliced = activity.time_slice(float(start), float(stop))
    np.testing.assert_array_equal(sliced.values[0], calcium[reference])
    np.testing.assert_array_equal(sliced.time.values, times[reference])
    assert sliced.clock == "calcium-native"
    updated = activity.with_metadata(selection="source-preserved")
    assert "selection" not in activity.metadata
    np.testing.assert_array_equal(updated.values, activity.values)
    assert updated.metadata["selection"] == "source-preserved"
    assert activity.select_signals([]).values.shape == (0, 1164)


def test_sparse_spike_public_operations_against_native_times(native):
    times, _, events = native
    support = [[float(times[0]), float(times[-1])]]
    direct = SpikeEvents(
        times=events,
        unit_indices=np.zeros(events.size, dtype=np.int64),
        signal_ids=["theis16-cell21"],
        observation_intervals=support,
        unit_observation_intervals={"theis16-cell21": support},
        signal_metadata={"source_cell": [21]},
        clock="paired-cell-attached",
    )
    labeled = SpikeEvents.from_unit_ids(
        times=events,
        unit_ids=["theis16-cell21"] * events.size,
        observation_intervals=support,
    )
    assert direct.n_signals == 1
    assert direct.n_events == 43
    np.testing.assert_array_equal(labeled.times, direct.times)
    np.testing.assert_array_equal(labeled.unit_indices, direct.unit_indices)
    selected = direct.select_signals("theis16-cell21")
    np.testing.assert_array_equal(selected.times, events)
    assert selected.signal_metadata["source_cell"] == (21,)
    start, stop = float(times[100]), float(times[900])
    expected = events[(events >= start) & (events < stop)]
    sliced = direct.time_slice(start, stop)
    np.testing.assert_array_equal(sliced.times, expected)
    assert sliced.observation_intervals.duration == pytest.approx(stop - start)
    assert sliced.unit_observation_intervals[
        "theis16-cell21"
    ].duration == pytest.approx(stop - start)
    quiet = direct.time_slice(float(times[0]), float(events[0]))
    assert quiet.n_events == 0
    assert quiet.n_signals == 1
    assert quiet.observation_intervals.duration == pytest.approx(events[0] - times[0])
    updated = direct.with_metadata(reference="source event timestamps")
    np.testing.assert_array_equal(updated.times, events)
    assert updated.metadata["reference"] == "source event timestamps"


def test_binned_container_against_independent_native_event_histogram(native):
    times, _, events = native
    edges = np.linspace(times[0], times[-1], 11)
    reference, _ = np.histogram(events, bins=edges)
    assert reference.sum() == 43
    assert reference.max() > 1
    activity = BinnedActivity(
        values=reference[None, :],
        bin_edges=edges,
        exposure=np.diff(edges),
        signal_ids=["theis16-cell21"],
        clock="paired-cell-attached",
        observation_intervals=[[float(edges[0]), float(edges[-1])]],
    )
    assert activity.n_bins == 10
    assert activity.n_signals == 1
    np.testing.assert_array_equal(activity.values[0], reference)
    np.testing.assert_allclose(activity.time.values, (edges[:-1] + edges[1:]) / 2)
    np.testing.assert_array_equal(activity.exposure[0], np.diff(edges))
    np.testing.assert_array_equal(activity.to_numpy()[0], reference)
    np.testing.assert_array_equal(activity.to_numpy(copy=False)[0], reference)
    np.testing.assert_array_equal(
        activity.select_signals("theis16-cell21").values[0], reference
    )
    np.testing.assert_array_equal(
        activity.select_bins(slice(2, 6)).values[0], reference[2:6]
    )
    selected_events = events[(events >= edges[2]) & (events < edges[6])]
    cut_reference, _ = np.histogram(selected_events, bins=edges[2:7])
    sliced = activity.time_slice(float(edges[2]), float(edges[6]))
    np.testing.assert_array_equal(sliced.values[0], cut_reference)
    np.testing.assert_array_equal(sliced.bin_edges, edges[2:7])
    np.testing.assert_array_equal(
        activity.with_metadata(oracle="numpy.histogram").values[0], reference
    )


def test_published_example_runs_as_a_script():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "examples/signals_from_published_data.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["calcium_shape"] == [1, 1164]
    assert summary["observed_spikes"] == summary["binned_spikes"] == 43
    assert summary["bins"] == 10
