"""Run: python examples/spike_rates_from_published_data.py (requires SciPy).

Uses unchanged published CASCADE DS01 AP timestamps. Observation support is
explicitly restricted to the imaging span; no fluorescence-to-spike inference.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mazepy.data import SpikeEvents
from mazepy.neural import bin_spikes, to_rate


def main():
    demo = Path(__file__).resolve().parent / "data/cascade_ds01"
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    path = demo / manifest["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Published fixture checksum mismatch.")
    source = loadmat(path, squeeze_me=True, struct_as_record=False)["CAttached"]
    record = np.asarray(source, dtype=object).reshape(-1)[0]
    time = np.asarray(record.fluo_time, dtype=np.float64).reshape(-1)
    events = np.asarray(record.events_AP, dtype=np.float64).reshape(-1) / 10_000
    start, stop = float(time[0]), float(time[-1])
    events = events[(events >= start) & (events < stop)]
    spikes = SpikeEvents.from_unit_ids(
        times=events,
        unit_ids=[21] * len(events),
        signal_ids=[21],
        observation_intervals=[[start, stop]],
        clock="paired-recording",
        metadata={"fixture": manifest["fixture_id"]},
    )
    edges = np.linspace(start, stop, 101)
    counts = bin_spikes(spikes, edges)
    rates = to_rate(counts)
    print(
        json.dumps(
            {
                "observed_spikes": spikes.n_events,
                "binned_spikes": int(counts.values.sum()),
                "bins": counts.n_bins,
                "rate_unit": rates.unit,
                "observation_duration_seconds": spikes.observation_intervals.duration,
                "spikes_reconstructed_from_rates": float(
                    np.nansum(rates.values * rates.exposure)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
