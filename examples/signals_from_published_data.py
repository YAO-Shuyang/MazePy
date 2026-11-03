"""Run from the repository root: python examples/signals_from_published_data.py."""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mazepy.data import BinnedActivity, CalciumTrace, SpikeEvents


def main():
    source_path = Path(__file__).resolve().parent / "data/cascade_ds01/cell21.mat"
    source = loadmat(source_path, squeeze_me=True, struct_as_record=False)["CAttached"]
    record = np.asarray(source, dtype=object).reshape(-1)[0]
    times = np.asarray(record.fluo_time, dtype=np.float64).reshape(-1)
    raw = np.asarray(record.fluo_mean, dtype=np.float64).reshape(1, -1)
    events = np.asarray(record.events_AP, dtype=np.float64).reshape(-1) / 10_000
    # Restrict explicitly to known imaging support; do not use the final spike
    # as an estimate of observation duration. All windows are half-open.
    start, stop = float(times[0]), float(times[-1])
    events = events[(events >= start) & (events < stop)]
    calcium = CalciumTrace(
        values=raw,
        time=times,
        signal_ids=["theis16-cell21"],
        unit="dF/F",
        signal_kind="dff",
        clock="paired-recording",
        metadata={"fixture": "cascade_ds01_theis16_cell21"},
    )
    spikes = SpikeEvents.from_unit_ids(
        times=events,
        unit_ids=["theis16-cell21"] * len(events),
        observation_intervals=[[start, stop]],
        clock="paired-recording",
    )
    # NumPy is an independent reference producer here; no MazePy binning
    # algorithm is implemented by these signal containers.
    edges = np.linspace(start, stop, 11)
    counts, _ = np.histogram(events, edges)
    binned = BinnedActivity(
        values=counts[None, :],
        bin_edges=edges,
        exposure=np.diff(edges),
        signal_ids=spikes.signal_ids,
        clock=spikes.clock,
        observation_intervals=spikes.observation_intervals,
    )
    print(
        json.dumps(
            {
                "calcium_shape": list(calcium.values.shape),
                "calcium_unit": calcium.unit,
                "observed_spikes": spikes.n_events,
                "binned_spikes": int(binned.values.sum()),
                "bins": binned.n_bins,
                "observation_duration_seconds": spikes.observation_intervals.duration,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
