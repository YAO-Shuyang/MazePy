# Published calcium/spike demo

This directory contains one unmodified real recording from the [CASCADE ground-truth database](https://github.com/HelmchenLabSoftware/Cascade/tree/5c5ccef3e45e1bf8b9bf0ae62a23a0f5d1550299/Ground_truth/DS01-OGB1-m-V1). The file has been renamed to `cell21.mat` for convenience; its bytes are unchanged.

- Original file: `Ground_truth/DS01-OGB1-m-V1/CAttached_Theis16_set2_OGB_V1_cell_21_mini.mat`.
- Source revision: `5c5ccef3e45e1bf8b9bf0ae62a23a0f5d1550299`.
- Size: **16,046 bytes**. SHA-256: `f79469ed0863a76e9756a56b88e19f2fa63dcbcdf7c3eccb2ca3dfb91f5ff8db`.
- Source modality: DS01, mouse V1, OGB-1 calcium imaging with simultaneous cell-attached action-potential timestamps.
- [Original immutable download](https://raw.githubusercontent.com/HelmchenLabSoftware/Cascade/5c5ccef3e45e1bf8b9bf0ae62a23a0f5d1550299/Ground_truth/DS01-OGB1-m-V1/CAttached_Theis16_set2_OGB_V1_cell_21_mini.mat).
- [Machine-readable provenance and limitations](manifest.json).

Cite the original [Theis et al., Neuron (2016)](https://doi.org/10.1016/j.neuron.2016.04.014), the [Berens et al., PLOS Computational Biology (2018)](https://doi.org/10.1371/journal.pcbi.1006157) benchmark referenced for DS01 by CASCADE, and the [Rupprecht et al., Nature Neuroscience (2021)](https://doi.org/10.1038/s41593-021-00895-5) database. The upstream repository's GPL-3.0 notice is retained as [LICENSE.upstream](LICENSE.upstream); this notice is separate from MazePy's code license.

## Native schema

`CAttached` contains one or more separate recordings from the same neuron. Each record has:

| Field | Meaning | Native units |
|---|---|---|
| `fluo_time` | Fluorescence sample times | Seconds |
| `fluo_mean` | ROI dF/F trace | dF/F |
| `events_AP` | Extracted action-potential times | 0.1 ms; divide by 10,000 for seconds |

Keep separate records' time supports separate. Do not infer the observation duration from the final spike. The mini file omits raw ephys voltage.

## Run the fixture checks

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/real_data -q -s
```

The tests verify integrity and scientific schema. They do not run a calcium-extraction pipeline or validate a MazePy neural algorithm, because those methods are not yet implemented. Per-method validation must call the method on relevant real data and compare against an independent expected result. This recording has no spatial behavior and cannot validate place/head-direction tuning or raw ephys/spike-sorting accuracy.
