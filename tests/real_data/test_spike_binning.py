"""Actual processing of native published AP times; independent reference counts.

The controlled-gap case is a declared subset of recorded events, not evidence
of a hardware dropout in the original recording. No synthetic events are added.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from mazepy.data import BinnedActivity, SpikeEvents
from mazepy.neural import bin_spikes, to_rate

pytestmark = pytest.mark.real_data
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples/data/cascade_ds01"


@pytest.fixture(scope="module")
def recorded():
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    path = DEMO / manifest["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    source = loadmat(path, squeeze_me=True, struct_as_record=False)["CAttached"]
    record = np.asarray(source, dtype=object).reshape(-1)[0]
    times = np.asarray(record.fluo_time, dtype=np.float64).reshape(-1)
    events = np.asarray(record.events_AP, dtype=np.float64).reshape(-1) / 10_000
    assert len(times) == 1164 and len(events) == 44
    # Imaging support is explicitly chosen; it is not inferred from last AP.
    events = events[(events >= times[0]) & (events < times[-1])]
    assert len(events) == 43
    return times, events


@pytest.mark.parametrize("layout", ["uniform", "irregular", "event_edges"])
def test_bin_spikes_on_native_recorded_events(recorded, layout):
    times, events = recorded
    if layout == "uniform":
        edges = np.linspace(times[0], times[-1], 101)
    elif layout == "irregular":
        edges = times[[0, 1, 13, 111, 300, 700, 900, 1163]]
    else:
        # A native event on the final right edge must be excluded, even though
        # numpy.histogram normally includes that edge in its final bin.
        edges = np.unique(np.r_[times[0], events[::5], events[-1]])
    spikes = SpikeEvents.from_unit_ids(
        times=events,
        unit_ids=[21] * len(events),
        signal_ids=[21],
        observation_intervals=[[float(times[0]), float(times[-1])]],
        metadata={"fixture": "cascade_ds01_theis16_cell21"},
        clock="paired",
    )
    selected = events[(events >= edges[0]) & (events < edges[-1])]
    reference, _ = np.histogram(selected, bins=edges)
    result = bin_spikes(spikes, edges)
    np.testing.assert_array_equal(result.values[0], reference)
    np.testing.assert_allclose(result.exposure[0], np.diff(edges), rtol=0, atol=1e-12)
    assert result.values.sum() == len(selected)
    assert result.signal_ids == (21,) and result.clock == "paired"
    assert result.valid.all()


def test_to_rate_on_independently_counted_recorded_spikes(recorded):
    times, events = recorded
    edges = times[[0, 100, 320, 601, 1000, 1163]]
    reference_counts, _ = np.histogram(events, bins=edges)
    reference_rate = np.asarray(
        [
            float(count) / float(right - left)
            for count, left, right in zip(
                reference_counts, edges[:-1], edges[1:], strict=True
            )
        ]
    )
    counts = BinnedActivity(
        values=reference_counts[None, :],
        bin_edges=edges,
        exposure=np.diff(edges),
        signal_ids=[21],
        observation_intervals=[[float(edges[0]), float(edges[-1])]],
    )
    rates = to_rate(counts)
    np.testing.assert_allclose(rates.values[0], reference_rate, rtol=1e-12, atol=1e-12)
    assert float(np.sum(rates.values * rates.exposure)) == pytest.approx(43, abs=1e-12)
    assert rates.unit == "Hz" and rates.signal_kind == "rate"


def test_processing_on_declared_subsets_of_recorded_spikes(recorded):
    times, events = recorded
    segments = [
        (float(times[0]), float(times[300])),
        (float(times[700]), float(times[-1])),
    ]
    selected = events[
        np.asarray([any(a <= t < b for a, b in segments) for t in events])
    ]
    spikes = SpikeEvents(
        times=selected,
        unit_indices=np.zeros(len(selected), dtype=int),
        signal_ids=[21],
        observation_intervals=[[float(times[0]), float(times[-1])]],
        unit_observation_intervals={21: segments},
    )
    edges = times[[0, 100, 300, 700, 1000, 1163]]
    reference_counts, _ = np.histogram(selected, bins=edges)
    reference_exposure = np.array(
        [
            sum(max(0.0, min(right, b) - max(left, a)) for a, b in segments)
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    result = bin_spikes(spikes, edges)
    np.testing.assert_array_equal(result.values[0], reference_counts)
    np.testing.assert_allclose(
        result.exposure[0], reference_exposure, rtol=0, atol=1e-12
    )
    rates = to_rate(result)
    assert reference_exposure[2] == 0
    assert not rates.valid[0, 2] and np.isnan(rates.values[0, 2])
    for i, seconds in enumerate(reference_exposure):
        if seconds > 0:
            assert rates.values[0, i] == pytest.approx(
                reference_counts[i] / seconds, abs=1e-12
            )
    assert np.nansum(rates.values * rates.exposure) == pytest.approx(
        len(selected), abs=1e-12
    )


def test_spike_rate_published_example_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/spike_rates_from_published_data.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["observed_spikes"] == summary["binned_spikes"] == 43
    assert summary["spikes_reconstructed_from_rates"] == pytest.approx(43, abs=1e-12)
    assert summary["bins"] == 100
