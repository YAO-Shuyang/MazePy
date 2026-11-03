# Spike counts and firing rates

`mazepy.neural` starts with two NumPy-only functions. They operate on the
[signal containers](SIGNALS.md); no native acquisition reader is required.

```python
from mazepy.data import SpikeEvents
from mazepy.neural import bin_spikes, to_rate

spikes = SpikeEvents.from_unit_ids(
    times=[0.1, 0.2, 0.2, 2.5],
    unit_ids=[21, 99, 21, 99],
    signal_ids=[21, 99, 105],
    observation_intervals=[[0.0, 1.0], [2.0, 4.0]],
    clock="ephys",
)
counts = bin_spikes(spikes, bin_edges=[0.0, 1.0, 2.0, 4.0])
rates = to_rate(counts)

print(counts.values)  # [[2, 0, 0], [1, 0, 1], [0, 0, 0]]
print(counts.exposure[0])  # [1., 0., 2.] seconds
print(rates.values)  # [[2., nan, 0.], [1., nan, 0.5], [0., nan, 0.]]
```

## `bin_spikes(spikes, bin_edges)`

- Input: `SpikeEvents` and explicitly supplied edges in seconds, in the same
  clock as the events. Edges must be finite and strictly increasing.
- Output: `BinnedActivity`, integer count matrix shaped `(units, bins)`,
  `signal_kind="count"`, `unit="spikes"`.
- Every bin is half-open `[left, right)`. A spike on an interior edge enters the
  next bin; a spike on the final right edge is excluded. Coincident spikes count
  separately. Events outside the requested range are ignored.
- Exposure is the intersection of each bin with the recording intervals. A
  unit's own intervals override the global support for that unit. Gaps contribute
  no time. Unequal bin widths and partial coverage are supported.
- A zero-exposure bin retains raw count zero and has `valid=False`. A fully or
  partially observed silent bin has `valid=True` and count zero.
- Signal order, silent units, clock, and source metadata are preserved. Global
  intervals are clipped to the requested window. Per-unit coverage is represented
  by the output exposure matrix; exact segment boundaries remain in the source
  `SpikeEvents` object.
- A single edge creates an empty `(units, 0)` result. Empty event vectors and
  empty unit rosters are supported.

Counting uses NumPy searches and integer accumulation, without a unit-by-event
comparison matrix. Storage of the result and its masks/exposure is dense, with
memory proportional to units times bins. This is an eager implementation;
chunked output and large-recording performance benchmarks remain future work.

## `to_rate(counts)`

Input must be `BinnedActivity` containing integer spike counts with
`signal_kind="count"`, `unit="spikes"`, and known `exposure` in seconds.
The result is `BinnedActivity` with float64 values, `signal_kind="rate"`, and
`unit="Hz"`:

\[
r_{ik} = \frac{c_{ik}}{e_{ik}}, \qquad e_{ik}>0.
\]

Valid observed silence becomes zero Hz. Invalid cells become NaN and remain
invalid. Nominal bin width is never substituted for missing exposure. Rates
outside finite float64 range raise an error. Exposure, bin edges, signal identity,
clock, and source metadata are retained. Source objects are not modified.

These are unsmoothed observed rates. No interpolation, synchronization, or
inference is applied. Calcium fluorescence, expected counts, and event weights
are different signal meanings and are not accepted by `to_rate`.

## Published-data example and tests

```bash
python -m pip install -e ".[dev]"
python examples/spike_rates_from_published_data.py
python -m pytest tests/neural tests/real_data/test_spike_binning.py -q
```

The example uses the unchanged [CASCADE DS01 recording](../examples/data/cascade_ds01/README.md).
Native action-potential timestamps are converted from 0.1 ms to seconds; explicit
restriction to imaging support retains 43 of the 44 native events. Binning into
100 intervals conserves all 43 selected events. Integrating rates over exposure
recovers those counts.

Method-specific tests compare counts against independently filtered
`numpy.histogram` results and rates against independently calculated count/time
ratios. An event-edge case checks final-edge exclusion. A controlled subset of
the same recorded events tests declared per-unit gaps; those exclusions are
imposed for the test and are not claimed to be native recording dropouts.
Analytical and seeded synthetic cases cover multiple units, silence, gaps,
partial exposure, malformed inputs, and numerical limits.

These checks establish deterministic counting and normalization on the included
recording. They do not measure spike-sorting accuracy, calcium inference,
statistical uncertainty, or generalization. See the maintained
[validation checklist](VALIDATION_CHECKLIST.md) for observed run evidence.
