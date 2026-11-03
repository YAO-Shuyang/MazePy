"""Boundary, identity, and metadata contracts for the shared containers."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mazepy.data import (
    BinnedActivity,
    CalciumTrace,
    NeuralRecording,
    SampledActivity,
    SpikeEvents,
)


def sampled(cls=SampledActivity, **changes):
    arguments = dict(
        values=np.array([[0.2, np.nan, 0.7, -0.1], [1, 2, 3, 4]], dtype=np.float32),
        time=[0.0, 0.1, 0.5, 0.9],
        unit="dF/F",
        signal_kind="dff",
        signal_ids=["roi-a", 27],
        clock="imaging",
        metadata={"source": {"files": ["source.mat"]}},
        signal_metadata={"depth": [10, 20]},
    )
    arguments.update(changes)
    return cls(**arguments)


@pytest.mark.parametrize("cls", [SampledActivity, CalciumTrace])
def test_sample_identity_masks_and_metadata_follow_selection(cls):
    activity = sampled(cls)
    assert isinstance(activity, NeuralRecording)
    assert activity.values.dtype == np.float32
    assert np.isnan(activity.values[0, 1])
    np.testing.assert_array_equal(activity.valid, [[1, 0, 1, 1], [1, 1, 1, 1]])
    selected = activity.select_signals([27, "roi-a"]).select_samples([0, 2])
    assert type(selected) is cls
    assert selected.signal_ids == (27, "roi-a")
    assert selected.signal_metadata["depth"] == (20, 10)
    np.testing.assert_array_equal(selected.time.values, [0, 0.5])
    np.testing.assert_allclose(selected.values, [[1, 3], [0.2, 0.7]])
    assert selected.clock == "imaging"
    assert selected.unit == "dF/F"
    assert selected.n_samples == 2
    assert selected.n_signals == 2
    assert activity.n_samples == 4


def test_half_open_slices_preserve_absolute_clock_and_unknown_support():
    activity = sampled()
    result = activity.time_slice(0.1, 0.9)
    np.testing.assert_array_equal(result.time.values, [0.1, 0.5])
    assert result.observation_intervals is None
    assert activity.time_slice(0.2, 0.2).values.shape == (2, 0)
    assert activity.select_signals([]).values.shape == (0, 4)


def test_declared_support_and_cell_specific_missingness():
    activity = sampled(observation_intervals=[[0, 0.2], [0.8, 1]])
    np.testing.assert_array_equal(activity.valid, [[1, 0, 0, 1], [1, 1, 0, 1]])
    cut = activity.time_slice(0.1, 0.95)
    np.testing.assert_allclose(
        cut.observation_intervals.values, [[0.1, 0.2], [0.8, 0.95]]
    )
    assert cut.observation_intervals.duration == pytest.approx(0.25)
    assert np.isnan(cut.values[0, 0])


def test_callers_and_returned_arrays_cannot_mutate_recording():
    data = np.ones((1, 3))
    metadata = {"settings": {"gain": [2]}}
    activity = sampled(
        values=data,
        time=[0, 1, 2],
        signal_ids=["a"],
        signal_metadata={"depth": [5]},
        metadata=metadata,
    )
    data[0, 0] = 999
    metadata["settings"]["gain"][0] = 999
    assert activity.values[0, 0] == 1
    assert activity.metadata["settings"]["gain"] == (2,)
    for array in (activity.values, activity.valid, activity.time.values):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        activity.metadata["new"] = 1
    with pytest.raises(FrozenInstanceError):
        activity.clock = "changed"
    changed = activity.with_metadata(analysis="test")
    assert "analysis" not in activity.metadata
    assert changed.metadata["analysis"] == "test"
    exported = activity.to_numpy()
    exported[:] = 12
    assert np.all(activity.values == 1)
    assert activity.to_numpy(copy=False) is activity.values


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"values": [1, 2, 3]}, ValueError),
        ({"time": [0, 1]}, ValueError),
        ({"time": [0, 1, 1, 2]}, ValueError),
        ({"signal_ids": [1, 1]}, ValueError),
        ({"signal_ids": [True, 2]}, TypeError),
        ({"signal_ids": "ab"}, TypeError),
        ({"signal_metadata": {"depth": [1]}}, ValueError),
        ({"clock": ""}, ValueError),
        ({"unit": ""}, ValueError),
        ({"signal_kind": "unknown"}, ValueError),
        ({"valid": [True, False]}, ValueError),
        ({"valid": [1, 0, 1, 0]}, TypeError),
        ({"values": np.ones((2, 4), dtype=complex)}, TypeError),
        ({"signal_kind": "count"}, TypeError),
        ({"signal_kind": "event_weight"}, ValueError),
    ],
)
def test_sampled_rejects_ambiguous_inputs(changes, error):
    with pytest.raises(error):
        sampled(**changes)


@pytest.mark.parametrize(
    "selection,error",
    [
        ([2, 1], ValueError),
        ([1, 1], ValueError),
        ([4], IndexError),
        ([-5], IndexError),
        ([0.1], TypeError),
        ([True], ValueError),
    ],
)
def test_invalid_sample_selection(selection, error):
    with pytest.raises(error):
        sampled().select_samples(selection)


def test_signal_selection_is_by_label_and_explicit():
    activity = sampled()
    assert activity.select_signals(27).signal_ids == (27,)
    with pytest.raises(KeyError):
        activity.select_signals([0])
    with pytest.raises(ValueError):
        activity.select_signals([27, 27])
    with pytest.raises(ValueError):
        CalciumTrace(values=[[1]], time=[0], unit="V", signal_kind="voltage")


@pytest.mark.parametrize(
    "kind, values, error",
    [
        ("binary", [[0, 2]], ValueError),
        ("probability", [[0, 1.1]], ValueError),
        ("probability", [[-0.1, 0.5]], ValueError),
        ("count", [[1.0, 2.0]], TypeError),
        ("count", [[1, -2]], ValueError),
        ("spike_events", [[1, 2]], ValueError),
    ],
)
def test_signal_meaning_is_checked(kind, values, error):
    with pytest.raises(error):
        SampledActivity(values=values, time=[0, 1], unit="a.u.", signal_kind=kind)


def test_event_weights_keep_fractional_amplitudes():
    activity = CalciumTrace(
        values=np.array([[0.02, 1.7, 0.003]], dtype=np.float32),
        time=[0, 0.05, 0.1],
        signal_kind="event_weight",
        unit="a.u.",
    )
    np.testing.assert_array_equal(
        activity.values, np.array([[0.02, 1.7, 0.003]], dtype=np.float32)
    )
    assert activity.values.dtype == np.float32


def spikes(**changes):
    arguments = dict(
        times=[0.1, 0.1, 0.5, 1.1],
        unit_indices=[0, 1, 0, 1],
        signal_ids=[11, "cluster-b", "silent"],
        observation_intervals=[[0, 2]],
        signal_metadata={"quality": ["good", "mua", "good"]},
        unit_observation_intervals={11: [[0, 1]], "cluster-b": [[0, 2]]},
    )
    arguments.update(changes)
    return SpikeEvents(**arguments)


def test_spike_labels_coincident_events_silent_units_and_selection():
    activity = spikes()
    assert activity.n_events == 4
    assert activity.n_signals == 3
    selected = activity.select_signals(["cluster-b", "silent", 11])
    np.testing.assert_array_equal(selected.times, [0.1, 0.1, 0.5, 1.1])
    np.testing.assert_array_equal(selected.unit_indices, [2, 0, 2, 0])
    assert selected.signal_metadata["quality"] == ("mua", "good", "good")
    silent = activity.select_signals("silent")
    assert silent.n_events == 0
    assert silent.n_signals == 1
    assert silent.observation_intervals.duration == 2
    assert activity.select_signals([]).n_signals == 0
    clipped = activity.time_slice(0.1, 1.1)
    assert clipped.n_events == 3
    assert clipped.observation_intervals.duration == pytest.approx(1)
    assert clipped.unit_observation_intervals[11].duration == pytest.approx(0.9)
    assert clipped.time_slice(0.2, 0.2).n_events == 0


def test_noncontiguous_source_cluster_ids_and_empty_input():
    activity = SpikeEvents.from_unit_ids(
        times=[0.1, 0.2, 0.2],
        unit_ids=[10, 999, 10],
        signal_ids=[999, 10, -5],
        observation_intervals=[[0, 1]],
    )
    assert activity.signal_ids == (999, 10, -5)
    np.testing.assert_array_equal(activity.unit_indices, [1, 0, 1])
    assert (
        SpikeEvents.from_unit_ids(
            times=[],
            unit_ids=[],
            observation_intervals=[[0, 1]],
        ).n_signals
        == 0
    )
    inferred = SpikeEvents.from_unit_ids(
        times=[0, 0],
        unit_ids=["b", "a"],
        observation_intervals=[[0, 1]],
    )
    assert inferred.signal_ids == ("b", "a")
    assert activity.with_metadata(source="test").metadata["source"] == "test"


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"observation_intervals": None}, ValueError),
        ({"times": [0.1, 0.2]}, ValueError),
        ({"times": [0.1, 0.2, 0.5, 2]}, ValueError),
        ({"times": [0.1, 0.2, 0.1, 1.1]}, ValueError),
        ({"times": np.array([2, 1, 3, 4], dtype=np.uint64)}, ValueError),
        ({"unit_indices": [0, 1, 0, 4]}, ValueError),
        ({"unit_indices": [0.0, 1.0, 0.0, 1.0]}, TypeError),
        ({"unit_observation_intervals": {11: [[0.2, 1]]}}, ValueError),
        ({"unit_observation_intervals": {11: [[0, 3]]}}, ValueError),
        ({"unit_observation_intervals": {"unknown": [[0, 1]]}}, ValueError),
        ({"unit": "Hz"}, ValueError),
    ],
)
def test_invalid_spike_contracts(changes, error):
    with pytest.raises(error):
        spikes(**changes)


def binned(**changes):
    arguments = dict(
        values=np.array([[0, 2, 5], [4, 0, 1]]),
        bin_edges=[0, 1, 3, 4],
        exposure=[1, 2, 1],
        signal_ids=["a", "b"],
        observation_intervals=[[0, 4]],
        signal_metadata={"depth": [10, 20]},
    )
    arguments.update(changes)
    return BinnedActivity(**arguments)


def test_binned_selection_counts_exposure_and_empty_cases():
    activity = binned()
    assert activity.n_bins == 3
    np.testing.assert_array_equal(activity.time.values, [0.5, 2, 3.5])
    selected = activity.select_signals(["b", "a"]).select_bins(slice(1, 3))
    np.testing.assert_array_equal(selected.values, [[0, 1], [2, 5]])
    np.testing.assert_array_equal(selected.exposure, [[2, 1], [2, 1]])
    assert selected.signal_metadata["depth"] == (20, 10)
    np.testing.assert_array_equal(activity.time_slice(1, 4).values, [[2, 5], [0, 1]])
    assert activity.time_slice(-10, 10).n_bins == 3
    assert activity.time_slice(4, 8).n_bins == 0
    assert activity.select_bins([]).exposure.shape == (2, 0)
    assert activity.select_signals([]).exposure.shape == (0, 3)
    assert activity.with_metadata(test=True).metadata["test"]
    assert activity.to_numpy().flags.writeable
    assert not activity.to_numpy(copy=False).flags.writeable


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"bin_edges": [0, 1, 2]}, ValueError),
        ({"bin_edges": [0, 1, 1, 2]}, ValueError),
        ({"exposure": [1, 3, 1]}, ValueError),
        ({"exposure": [1, -1, 1]}, ValueError),
        ({"exposure": [1, float("nan"), 1]}, ValueError),
        ({"exposure": [1, 1]}, ValueError),
    ],
)
def test_invalid_binned_contracts(changes, error):
    with pytest.raises(error):
        binned(**changes)


def test_zero_exposure_and_partial_bins_are_explicit():
    activity = binned(exposure=[[1, 0, 1], [0.5, 1, 0]])
    np.testing.assert_array_equal(activity.valid, [[1, 0, 1], [1, 1, 0]])
    with pytest.raises(ValueError, match="cuts a bin"):
        activity.time_slice(0.5, 3)
    with pytest.raises(ValueError, match="contiguous"):
        activity.select_bins([0, 2])
    assert binned(exposure=None).exposure is None


def test_binned_exposure_respects_observation_gaps():
    activity = binned(
        observation_intervals=[[0, 0.5], [3, 4]],
        exposure=[0.5, 0, 1],
    )
    np.testing.assert_array_equal(activity.valid, [[1, 0, 1], [1, 0, 1]])
    with pytest.raises(ValueError, match="observation support"):
        binned(observation_intervals=[[0, 0.5], [3, 4]], exposure=[1, 2, 1])
    assert (
        not binned(
            observation_intervals=[[0, 0.5], [3, 4]],
            exposure=None,
        )
        .valid[:, 1]
        .any()
    )
    masked = binned(values=np.array([[-1, 2, 5], [4, 0, 1]]), exposure=[0, 2, 1])
    assert not masked.valid[0, 0]


def test_huge_unmapped_indices_cannot_allocate_a_roster():
    with pytest.raises(ValueError, match="Non-contiguous"):
        SpikeEvents(
            times=[0.1],
            unit_indices=np.array([2**64 - 1], dtype=np.uint64),
            observation_intervals=[[0, 1]],
        )


def test_empty_boolean_mask_still_checks_axis_length():
    with pytest.raises(ValueError):
        sampled().select_samples(np.array([], dtype=bool))


def test_rates_need_explicit_units_and_keep_float_values():
    with pytest.raises(ValueError, match="units explicitly"):
        BinnedActivity(values=[[0.2, 0.7]], bin_edges=[0, 1, 2], signal_kind="rate")
    rates = BinnedActivity(
        values=[[0.2, 0.7]],
        bin_edges=[0, 1, 2],
        signal_kind="rate",
        unit="Hz",
    )
    np.testing.assert_array_equal(rates.values, [[0.2, 0.7]])
    assert rates.unit == "Hz"


def test_empty_bin_slices_keep_the_requested_absolute_boundary():
    activity = binned()
    np.testing.assert_array_equal(activity.select_bins(slice(2, 2)).bin_edges, [3])
    np.testing.assert_array_equal(activity.select_bins(slice(10, 10)).bin_edges, [4])
    assert activity.select_bins(slice(2, 2)).values.shape == (2, 0)


def test_extreme_bin_width_overflow_is_rejected():
    with pytest.raises(ValueError, match="finite bin durations"):
        BinnedActivity(values=np.array([[1]]), bin_edges=[-1e308, 1e308])
