"""Analytical and seeded independent examples for spike counting and rates."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mazepy.data import BinnedActivity, SpikeEvents
from mazepy.neural import bin_spikes, to_rate


def test_half_open_bins_coincident_spikes_silent_units_and_identity():
    spikes = SpikeEvents.from_unit_ids(
        times=[-0.5, 0, 0.25, 1, 1, 2, 3, 4, 4.5],
        unit_ids=[41, 9, 41, 41, 41, 9, 41, 41, 9],
        signal_ids=[41, 9, 105],
        observation_intervals=[[-1, 5]],
        clock="ephys",
        metadata={"source": "analytical example"},
        signal_metadata={"quality": ["good", "mua", "good"]},
    )
    counts = bin_spikes(spikes, [0, 1, 2, 4])
    np.testing.assert_array_equal(counts.values, [[1, 2, 1], [1, 0, 1], [0, 0, 0]])
    assert counts.values.dtype == np.int64
    np.testing.assert_array_equal(counts.exposure, [[1, 1, 2]] * 3)
    np.testing.assert_array_equal(counts.observation_intervals.values, [[0, 4]])
    assert counts.valid.all()
    assert counts.signal_ids == spikes.signal_ids
    assert counts.clock == "ephys"
    assert counts.metadata == spikes.metadata
    assert counts.signal_metadata == spikes.signal_metadata
    rates = to_rate(counts)
    np.testing.assert_array_equal(rates.values, [[1, 2, 0.5], [1, 0, 0.5], [0, 0, 0]])
    assert rates.signal_kind == "rate" and rates.unit == "Hz"
    assert counts.signal_kind == "count" and counts.unit == "spikes"
    assert rates.signal_ids == spikes.signal_ids
    assert rates.metadata == spikes.metadata
    assert rates.signal_metadata == spikes.signal_metadata
    assert rates.clock == spikes.clock
    np.testing.assert_array_equal(rates.bin_edges, counts.bin_edges)
    np.testing.assert_array_equal(rates.exposure, counts.exposure)
    np.testing.assert_array_equal(rates.valid, counts.valid)
    assert spikes.n_events == 9
    for result in (counts, rates):
        with pytest.raises(ValueError):
            result.values[0, 0] = 10
        with pytest.raises(FrozenInstanceError):
            result.clock = "other"


def test_recording_gaps_partial_bins_and_per_unit_support():
    spikes = SpikeEvents(
        times=[0.25, 0.75, 2, 3.5],
        unit_indices=[0, 0, 0, 0],
        signal_ids=["active", "silent", "absent"],
        observation_intervals=[[0, 1], [2, 4]],
        unit_observation_intervals={"active": [[0.25, 1], [2, 3.75]], "absent": []},
    )
    counts = bin_spikes(spikes, [-1, 0, 0.5, 1, 2, 3, 5])
    np.testing.assert_array_equal(counts.values[0], [0, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(counts.values[1:], np.zeros((2, 6)))
    exposure = [[0, 0.25, 0.5, 0, 1, 0.75], [0, 0.5, 0.5, 0, 1, 1], [0] * 6]
    np.testing.assert_array_equal(counts.exposure, exposure)
    np.testing.assert_array_equal(counts.valid, np.asarray(exposure) > 0)
    rates = to_rate(counts)
    np.testing.assert_allclose(rates.values[0], [np.nan, 4, 2, np.nan, 1, 4 / 3])
    np.testing.assert_allclose(rates.values[1], [np.nan, 0, 0, np.nan, 0, 0])
    assert np.isnan(rates.values[2]).all()
    # Reordering source units must reorder the computed matrix/exposure as well.
    reordered = bin_spikes(
        spikes.select_signals(["absent", "active"]), counts.bin_edges
    )
    np.testing.assert_array_equal(reordered.values, counts.values[[2, 0]])
    np.testing.assert_array_equal(reordered.exposure, counts.exposure[[2, 0]])


@pytest.mark.parametrize("roster", [[], ["silent"]])
@pytest.mark.parametrize("edges", [[2], [0, 1, 2]])
@pytest.mark.parametrize("support", [[], [[0, 1]]])
def test_empty_units_events_support_and_bins(roster, edges, support):
    spikes = SpikeEvents(
        times=[], unit_indices=[], signal_ids=roster, observation_intervals=support
    )
    counts = bin_spikes(spikes, edges)
    assert counts.values.shape == (len(roster), len(edges) - 1)
    assert not counts.values.any()
    np.testing.assert_array_equal(counts.valid, counts.exposure > 0)
    rates = to_rate(counts)
    assert rates.values.shape == counts.values.shape
    np.testing.assert_array_equal(rates.valid, counts.valid)
    assert (rates.values[rates.valid] == 0).all()
    assert np.isnan(rates.values[~rates.valid]).all()


def test_multiple_segments_in_one_bin_and_small_durations():
    spikes = SpikeEvents(
        times=[0, 2, 9],
        unit_indices=[0] * 3,
        observation_intervals=[[0, 1], [2, 3], [9, 10]],
    )
    counts = bin_spikes(spikes, [0, 10])
    np.testing.assert_array_equal(counts.exposure, [[3]])
    np.testing.assert_array_equal(to_rate(counts).values, [[1]])
    tiny = SpikeEvents(
        times=[5e-11, 9e-10],
        unit_indices=[0, 0],
        observation_intervals=[[-1e15, 1]],
    )
    counts = bin_spikes(tiny, [0, 1e-10, 1e-9])
    np.testing.assert_array_equal(counts.values, [[1, 1]])
    np.testing.assert_allclose(counts.exposure, [[1e-10, 9e-10]], rtol=1e-15, atol=0)


def test_seeded_irregular_example_against_direct_event_and_overlap_loops():
    rng = np.random.default_rng(20261103)
    support = np.array([[-3, -1.5], [0, 1.25], [1.5, 4], [5, 8]])
    roster = ["a", "b", "c", "silent"]
    per_unit = {"b": support[[0, 2]], "c": [[0.25, 1], [6, 7]]}
    event_rows = []
    for row, label in enumerate(roster[:-1]):
        for start, stop in per_unit.get(label, support):
            event_rows.extend((time, row) for time in rng.uniform(start, stop, 25))
    event_rows.sort()
    times, indices = np.asarray(event_rows).T
    spikes = SpikeEvents(
        times=times,
        unit_indices=indices.astype(int),
        signal_ids=roster,
        observation_intervals=support,
        unit_observation_intervals=per_unit,
    )
    edges = np.sort(np.r_[-4, rng.uniform(-3.5, 8.5, 80), 9])
    counts = bin_spikes(spikes, edges)
    expected_counts = np.zeros(counts.values.shape, dtype=int)
    expected_exposure = np.zeros(counts.values.shape)
    for row, label in enumerate(roster):
        for column, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            expected_counts[row, column] = sum(
                event_row == row and left <= time < right
                for time, event_row in event_rows
            )
            expected_exposure[row, column] = sum(
                max(0.0, min(right, stop) - max(left, start))
                for start, stop in per_unit.get(label, support)
            )
    np.testing.assert_array_equal(counts.values, expected_counts)
    np.testing.assert_allclose(counts.exposure, expected_exposure, rtol=0, atol=1e-14)
    assert counts.values.sum() == spikes.n_events
    rates = to_rate(counts)
    np.testing.assert_allclose(
        (rates.values * counts.exposure)[rates.valid],
        expected_counts[rates.valid],
        rtol=1e-14,
        atol=1e-14,
    )


def test_to_rate_preserves_masks_and_never_uses_invalid_raw_counts():
    counts = BinnedActivity(
        values=[[4, -99, 0, 3]],
        bin_edges=[0, 2, 3, 4, 5],
        exposure=[2, 1, 0, 1],
        valid=[True, False, True, True],
    )
    rates = to_rate(counts)
    np.testing.assert_allclose(rates.values, [[2, np.nan, np.nan, 3]])
    np.testing.assert_array_equal(rates.valid, [[True, False, False, True]])
    np.testing.assert_array_equal(counts.values, [[4, -99, 0, 3]])


@pytest.mark.parametrize(
    ("edges", "error"),
    [
        ([], ValueError),
        ([[0, 1]], ValueError),
        ([0, 0], ValueError),
        ([1, 0], ValueError),
        ([0, np.nan], ValueError),
        ([0, np.inf], ValueError),
        ([-1e308, 1e308], ValueError),
        (["0", "1"], TypeError),
        ([0j, 1j], TypeError),
        ([False, True], TypeError),
    ],
)
def test_invalid_bin_edges_are_rejected(edges, error):
    spikes = SpikeEvents(times=[], unit_indices=[], observation_intervals=[])
    with pytest.raises(error):
        bin_spikes(spikes, edges)


@pytest.mark.parametrize("function", [bin_spikes, to_rate])
def test_wrong_recording_type_is_rejected(function):
    with pytest.raises(TypeError):
        function(None, [0, 1]) if function is bin_spikes else function(None)


def test_to_rate_requires_explicit_exposure_even_with_known_recording_support():
    counts = BinnedActivity(
        values=[[2]], bin_edges=[0, 1], observation_intervals=[[0, 1]]
    )
    with pytest.raises(ValueError, match="exposure"):
        to_rate(counts)


@pytest.mark.parametrize(
    ("kind", "unit", "values"),
    [
        ("rate", "Hz", [[1.0]]),
        ("expected_count", "spikes", [[1.5]]),
        ("binary", "spikes", [[1]]),
        ("count", "photons", [[2]]),
    ],
)
def test_to_rate_rejects_other_signal_meanings(kind, unit, values):
    source = BinnedActivity(
        values=values,
        bin_edges=[0, 1],
        exposure=[1],
        signal_kind=kind,
        unit=unit,
    )
    with pytest.raises(ValueError, match="spike counts"):
        to_rate(source)


def test_rate_overflow_is_an_error():
    counts = BinnedActivity(values=[[1]], bin_edges=[0, 1], exposure=[5e-324])
    with pytest.raises(ValueError, match="float64"):
        to_rate(counts)
