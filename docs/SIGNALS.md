# Signal containers

`mazepy.data` supplies shared, NumPy-backed structures for recorded and simulated
activity. They are dataclasses containing arrays. Analysis functions and native
pipeline importers are separate parts of the implementation plan.

| Type | Content | Important fields |
|---|---|---|
| `NeuralRecording` | Abstract shared identity/selection interface | IDs, units, kind, clock, metadata, observation intervals |
| `TimeAxis` | Explicit, increasing sample timestamps in seconds | `values` |
| `TimeIntervals` | Sorted, non-overlapping observation segments | `values` shaped `(segments, 2)` |
| `SampledActivity` | Real-valued samples shaped `(signals, samples)` | `values`, `time`, `valid` |
| `CalciumTrace` | A sampled calcium representation | Explicit `signal_kind` and units |
| `SpikeEvents` | Sparse event times and row indices into a unit roster | `times`, `unit_indices`, `signal_ids` |
| `BinnedActivity` | Counts, rates, or other already-binned values | `values`, `bin_edges`, optional `exposure` |

## Sampled signals

```python
import numpy as np
from mazepy.data import CalciumTrace, SampledActivity, TimeAxis

time = TimeAxis.regular(4, sampling_rate=20)
activity = CalciumTrace(
    values=np.array(
        [[0.02, 0.5, np.nan, -0.01], [0.1, 0.2, 0.4, 0.3]], dtype=np.float32
    ),
    time=time,
    unit="dF/F",
    signal_kind="dff",
    signal_ids=["roi-a", "roi-b"],
    clock="imaging",
    signal_metadata={"plane": [0, 0]},
    metadata={"source": "session-001"},
)

subset = activity.select_signals(["roi-b", "roi-a"])
segment = activity.time_slice(0.05, 0.15)
print(segment.values.shape)  # (2, 2): [0.05, 0.15)
```

Floating-point dtype and amplitudes are retained. A non-finite sample is invalid
for that signal only: no column is deleted and no NaN is replaced by zero. Supply
`valid` as `(samples,)` or `(signals, samples)` to mark additional exclusions.

`CalciumTrace` defaults to raw `fluorescence`; choose `dff`, `calcium`,
`corrected_fluorescence`, `event_weight`, `expected_count`, `rate`, `probability`,
or `binary` when that is what the input represents. Units are required. The class
does not calculate a baseline, deconvolve a signal, infer spike counts, or apply
an event threshold.

`SampledActivity` provides the same storage contract for other sampled signals,
including `voltage`, `optical_voltage`, `current`, `phase`, `amplitude`, and
`power`, with explicitly supplied units. Hardware calibration and native voltage
or photometry acquisition readers are separate capabilities. Published-data
validation currently uses the calcium/spike fixture described below.

## Sparse spikes

```python
from mazepy.data import SpikeEvents

spikes = SpikeEvents.from_unit_ids(
    times=[0.1, 0.2, 0.2, 1.7],
    unit_ids=[21, 99, 21, 99],
    signal_ids=[21, 99, 105],
    observation_intervals=[[0.0, 10.0]],
    signal_metadata={"quality": ["good", "mua", "good"]},
    clock="ephys",
)
print(spikes.unit_indices)  # [0, 1, 0, 1]
quiet_unit = spikes.select_signals(105)
assert quiet_unit.n_events == 0
assert quiet_unit.observation_intervals.duration == 10.0
```

Direct construction uses `times` plus integer `unit_indices` into `signal_ids`.
Use `from_unit_ids` for arbitrary source labels. A supplied roster preserves
silent units. Without one, direct indices must be contiguous starting at zero;
source cluster numbers are never treated as a giant dense matrix dimension.

Times are non-decreasing, allowing coincident events. They must lie in declared
observation support. No duration is inferred from the last spike. Optional
`unit_observation_intervals` maps individual unit IDs to narrower observation
segments; these constrain events and follow unit/time selection.

## Time and observation support

- All time values and intervals use seconds. Preserve the original clock's
  origin; a time slice does not reset it to zero.
- Windows and observation intervals are half-open: `[start, stop)`.
- Sample timestamps must be finite and strictly increasing. Spike times can
  coincide. Neither vector is silently sorted or cleaned.
- `TimeAxis.regular` constructs an explicit timestamp vector. It is a generator,
  rather than a synchronization or resampling algorithm.
- A sampled recording's `observation_intervals=None` means unknown exposure.
  Sample centers alone do not define the final frame's duration.
- `time_slice` intersects any declared intervals with the requested window.
  `select_samples` selects ordered sample points and retains source support as
  context. It does not assert continuous valid coverage between selected points.
- Independently named clocks require an explicit synchronization mapping before
  cross-stream analysis. Sharing a string label does not estimate a mapping.

## Already-binned activity

```python
from mazepy.data import BinnedActivity

counts = BinnedActivity(
    values=[[0, 2, 5]],
    bin_edges=[0.0, 1.0, 3.0, 4.0],
    exposure=[1.0, 1.5, 1.0],
    observation_intervals=[[0.0, 4.0]],
    signal_ids=[21],
)
middle = counts.time_slice(1.0, 3.0)
assert middle.values[0, 0] == 2
```

Count data use an integer dtype and can exceed one. Default count units are
`spikes`; choose another unit explicitly when appropriate. Non-count binned
signals require explicit units, such as `unit="Hz", signal_kind="rate"`.

`bin_edges` has length `n_bins + 1`. `time` returns bin centers. Optional exposure
is valid observation time in seconds, shaped `(bins,)` or `(signals, bins)`;
unknown exposure remains `None`. Exposure cannot exceed the bin's overlap with
declared observation support. Zero exposure marks a cell invalid while retaining
its raw buffer value. Supplied masks and source NaNs are retained independently.

`select_bins` accepts a contiguous range. `time_slice` must align to bin edges;
it raises an error if a boundary cuts a bin. This prevents a whole-bin count or
mean from being relabeled as a partial-bin measurement. Rebin original events or
samples when different edges are needed. Binning and rate-estimation algorithms
are not implemented by this container.

## Identity, metadata, and memory

`select_signals` selects by labels, not row positions, and can reorder rows.
Selection preserves labels, valid masks, clock, unit, kind, and aligned
`signal_metadata`. Empty selections retain the other axis and declared support.
An all-silent event slice retains its signal roster.

Construction makes defensively owned, read-only numerical arrays. Metadata is
frozen recursively. Methods return new objects; `with_metadata(**updates)`
changes provenance in a new object. Use `to_numpy()` for a writable copy, or
`to_numpy(copy=False)` for the stored read-only matrix.

This first implementation is eager and dense for sampled/binned matrices. Spike
events remain sparse. Lazy raw-acquisition adapters, session bundles, file
serialization, and processed result containers will be added in later steps.

## Published-data checks

```bash
python -m pip install -e ".[dev]"
python examples/signals_from_published_data.py
python -m pytest tests/data tests/real_data/test_signal_containers.py -q
```

The [pinned CASCADE DS01 fixture](../examples/data/cascade_ds01/README.md) contains
1,164 calcium samples and 44 native cell-attached action-potential timestamps.
Explicit restriction to first-to-last imaging sample support retains 43 spikes.
Tests invoke each concrete public container API and compare selection/storage
results to direct SciPy fields, independent timestamp masks, and a NumPy event
histogram. The regular-axis generator is additionally checked with fixture-derived
dimensions/rate; it does not replace the native clock. Synthetic cases cover
gaps, multi-unit remapping, zero exposure, immutable buffers, and rejected inputs.

This evidence establishes the shared data contracts on the included recording.
It does not establish native pipeline-reader compatibility, inference accuracy,
voltage calibration, spatial tuning, or broad biological generalization. See
the [per-method checklist](VALIDATION_CHECKLIST.md) for recorded outcomes.
