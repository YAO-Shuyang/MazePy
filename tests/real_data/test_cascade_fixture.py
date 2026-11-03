"""Integrity/schema checks for an unmodified published recorded-data fixture.

These checks do not establish validation of any MazePy neural algorithm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

DEMO = Path(__file__).resolve().parents[2] / "examples/data/cascade_ds01"
pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def source_manifest():
    return json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recorded_segments(source_manifest):
    content = loadmat(
        DEMO / source_manifest["file"], squeeze_me=True, struct_as_record=False
    )
    assert "CAttached" in content
    return np.asarray(content["CAttached"], dtype=object).reshape(-1)


def test_published_fixture_integrity(source_manifest):
    data = (DEMO / source_manifest["file"]).read_bytes()
    assert len(data) == source_manifest["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == source_manifest["sha256"]
    assert source_manifest["data_origin"] == "recorded"
    assert len(source_manifest["source"]["commit"]) == 40
    assert source_manifest["source"]["commit"] in source_manifest["source"]["url"]
    assert source_manifest["publications"]
    assert (DEMO / source_manifest["source_license"]["file"]).is_file()


def test_recorded_calcium_spike_schema(recorded_segments, source_manifest):
    assert len(recorded_segments) > 0
    assert source_manifest["schema"]["time_unit"] == "s"
    assert source_manifest["schema"]["events_to_seconds"] == 0.0001
    summaries = []
    overlapping_events = 0
    for index, segment in enumerate(recorded_segments):
        time = np.asarray(segment.fluo_time, dtype=np.float64).reshape(-1)
        signal = np.asarray(segment.fluo_mean, dtype=np.float64).reshape(-1)
        native_events = np.asarray(segment.events_AP, dtype=np.float64).reshape(-1)
        assert time.size == signal.size
        assert time.size >= 2
        assert np.isfinite(time).all()
        assert np.isfinite(signal).all()
        assert np.isfinite(native_events).all()
        assert np.all(np.diff(time) > 0)
        assert np.all(np.diff(native_events) >= 0)
        assert np.ptp(signal) > 0
        seconds = native_events / 10_000.0
        duration = float(time[-1] - time[0])
        assert duration > 0
        in_support = int(
            np.count_nonzero((seconds >= time[0]) & (seconds <= time[-1]))
        )
        overlapping_events += in_support
        summaries.append(
            {
                "record": index,
                "samples": int(time.size),
                "native_spikes": int(seconds.size),
                "spikes_in_imaging_support": in_support,
                "observed_time_span_seconds": duration,
                "median_sample_interval_seconds": float(np.median(np.diff(time))),
            }
        )
    assert overlapping_events > 0
    print("RECORDED_FIXTURE_SUMMARY=" + json.dumps(summaries, sort_keys=True))
