# MazePy neural data: support and implementation plan

Research checked: November 2, 2026. All MazePy interfaces, milestones, and support tiers in this document are proposals, not implemented capabilities. Upstream compatibility must be checked against the versions selected during implementation.

MazePy should provide a common route from cellular activity and electrical recordings to behavior-aligned population activity, tuning curves, and spatial analyses. Broad coverage is best achieved through a small internal model, dedicated readers for frequently used processed outputs, and adapters to established format libraries. Acquisition hardware, file format, processing method, and signal meaning must remain separate concepts.

The initial focus should be Suite2p and CaImAn/CNMF-E imaging, sorted extracellular spikes including Kilosort/Phy, generic laboratory arrays, and NWB. Raw extracellular recordings should be accessible in chunks through SpikeInterface. Other supported backends and modalities can then expand coverage without changing the analysis API.

## 1. Repositories and designs to study

The following is a practical interoperability survey, not a quantitative ranking of adoption. Repository links identify the relevant projects; the sources at the end document the capabilities used in this plan.

| Repository | Relevant purpose | Proposed relationship to MazePy |
|---|---|---|
| [Pynapple](https://github.com/pynapple-org/pynapple) | Timestamped neural data, intervals, tuning, and time-series analysis | Closest reference for shared analysis interfaces; optional object conversion and numerical comparisons. [S1] |
| [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) | Raw recording and sorting readers, sorting execution, curation, and quality analysis | Main electrophysiology backend; accept its recording and sorting objects directly. [S2, S3] |
| [Neo](https://github.com/NeuralEnsemble/python-neo) | Electrophysiology objects and vendor readers | Optional conversion for analog signals, spikes, events, and segments; extend hardware coverage. [S4] |
| [Elephant](https://github.com/NeuralEnsemble/elephant) | Spike-train and electrophysiology analyses | Optional specialist analyses and reference checks for overlapping calculations. [S5] |
| [PyNWB](https://github.com/NeurodataWithoutBorders/pynwb) | Python interface to NWB | Main standardized scientific exchange route for supported NWB objects. [S6, S7] |
| [NeuroConv](https://github.com/catalystneuro/neuroconv) | Conversion of acquisition and processing formats to NWB | Optional conversion route for less common formats; useful reference for readers and fixtures. [S11] |
| [Suite2p](https://github.com/MouseLand/suite2p) | Calcium extraction and processing | Dedicated reader for output folders and MATLAB exports; preserve neuropil and ROI metadata. [S8] |
| [CaImAn](https://github.com/flatironinstitute/CaImAn) | Calcium processing, CNMF/CNMF-E, online methods, and VolPy | Dedicated calcium result reader and a voltage-output adapter. [S9, S14] |
| [MiniAn](https://github.com/miniscope/minian) | Calcium processing with labeled arrays and chunked storage | Reader for its xarray/Zarr output; preserve labels and lazy access. The current repository is under Miniscope. [S10, S23] |
| [EXTRACT](https://github.com/schnitzer-lab/EXTRACT-public) | Calcium trace extraction | Dedicated MATLAB-output mapping after the initial readers. [S13] |
| [CASCADE](https://github.com/HelmchenLabSoftware/Cascade) | Learned calcium-to-spike inference | Import predictions with explicit expected-count versus rate semantics. [S15] |
| [CellExplorer](https://github.com/petersenpeter/CellExplorer) | Sorted-unit characterization and hippocampal laboratory workflows | Import spikes and optional cell metadata from its documented MATLAB structures. [S18] |
| [MNE-Python](https://github.com/mne-tools/mne-python) | EEG/MEG and related electrophysiology | Later optional adapter for continuous/epoched data and device formats. [S17] |

MazePy's particular contribution should be the connection between these data, maze geometry, accessible-space binning, wall-aware smoothing, route analysis, and comparison with simulations. Generic file readers should be reused where practical.

## 2. Meaning of support

Every reader should have a published status, backend, tested source versions, and a capability record. The existence of an upstream reader is insufficient to claim a format works in MazePy.

| Status | Meaning |
|---|---|
| Native, tested | MazePy owns the reader or output-schema mapping and verifies it against representative fixtures. |
| Backend, tested | MazePy's adapter and a specific upstream reader/version have passed integration checks. |
| Backend, available | The upstream library advertises a reader; MazePy compatibility has not yet been established. |
| Mapped arrays | The user supplies array keys, units, orientation, timestamps, and signal meaning; no pipeline-specific interpretation is promised. |
| Planned | A reader or workflow appears on this roadmap but is not available yet. |

Support also has separate capabilities: trace access, spike events, timestamps, digital events, ROI footprints, electrode geometry, curation labels, quality metrics, lazy access, and export. Readers should report these independently. For example, an importer may correctly read spike times without supporting raw waveform access.

## 3. Neural activity to represent

| Signal family | Core representation | Essential distinctions | Priority |
|---|---|---|---|
| Extracellular sorted spikes | `SpikeEvents`: timestamps and unit IDs | Single units versus multiunit; curated IDs; per-unit observation intervals; sample indices versus seconds | First |
| Binned spikes/events | `BinnedActivity`: N by K values plus edges | Integer counts may exceed one; binary occupancy is distinct from counts | First |
| Calcium fluorescence | `CalciumTrace`, based on `SampledActivity` | Raw F, neuropil F, corrected F, baseline, dF/F, denoised calcium, and deconvolved event weights remain named streams | First |
| Inferred neural activity | Sampled expected counts, event weights, probabilities, or rates | Preserve floating values, estimator identity, units, and calibration; conversion must be explicit | First |
| Raw extracellular voltage | Lazy `ElectricalTrace`/`SampledActivity` | Physical scaling, channel order, probe contacts, multiple streams, and recording segments | Second |
| LFP and derived signals | `LFPTrace` and sampled derived signals | Voltage, phase, amplitude, and band power have different units and transformations | Second |
| Voltage imaging | `VoltageTrace` plus optional `SpikeEvents` | Optical signal versus membrane voltage; polarity, subthreshold signal, sampling rate, and detection provenance | Second |
| Fiber photometry | `PhotometryTrace` | Fiber/region signals; signal and reference channels; isosbestic/reference roles | Later |
| Intracellular/patch clamp | Electrical signals with sweep and stimulus metadata | Current clamp versus voltage clamp; stimulus traces; voltage/current units | Later |
| EEG, ECoG, and MEG | Continuous or epoched channel signals | Sensor types and locations; montage/reference; annotations and bad intervals | Later |
| Simulated neural populations | Any corresponding core representation | Simulation step, population IDs, observation model, and ground truth | First through generic arrays |

One-photon, two-photon, and volumetric imaging can use the same extracted-trace model, with plane/volume/channel metadata and separate clocks where required. Neuropixels and tetrodes can use the same spike-event model. Multiunit threshold crossings must retain their multiunit identity.

Raw microscopy movies have pixel and image axes. They should be linked as source assets or passed to an established extraction pipeline. Region- or cell-extracted widefield signals can use the sampled-signal model. Imaging source extraction is a distinct workflow from importing processed neural activity.

## 4. Initial imaging interfaces

| Workflow | Input to recognize | Content to retain | Route and priority |
|---|---|---|---|
| Suite2p | Plane folders with `F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`; `Fall.mat` | All trace representations, original ROI indices, cell classification, spatial masks, plane/channel metadata | Dedicated reader in first release. Test classic `ops.npy` layouts and newer metadata layouts separately. [S8] |
| CaImAn CNMF/CNMF-E/OnACID | Saved CNMF HDF5 result; in-memory estimates | Available `A`, `C`, `S`, `YrA`, `F_dff`, accepted/rejected component indices, parameters | Dedicated reader in first release. Fields can be absent. Keep C and S distinct. [S9] |
| MATLAB CNMF-E | Documented CNMF-E `.mat` results | Named temporal/spatial components and available quality/selection metadata | First release for a verified representative schema; field mapping for other variants. NeuroConv provides an established conversion route. [S12] |
| MiniAn | xarray-augmented Zarr output; exported NetCDF/xarray datasets | Temporal arrays, spatial footprints, unit labels, frame coordinates | Second stage; retain lazy arrays and match components by labels. [S10] |
| EXTRACT | MATLAB analysis output | Temporal activity, spatial weights, available selection/quality metadata | Second stage with fixtures; inspect dimensions and sampling metadata. [S13] |
| Inscopix | `.isxd` cell/event sets; exported CSV plus footprints | Traces/events, cell names/status, timing, footprints | Second stage, using optional `isx` when available; exported-array route also available. [S16] |
| MIN1PIPE | Documented refined MATLAB output, including `sigfn` | Processed traces and corresponding ROI information | Later dedicated mapping; generic MATLAB route can precede it. [S22] |
| CASCADE | Returned prediction arrays or `.npy`/`.mat` exports | Prediction values, frame rate, model and preprocessing metadata | Second stage; output is expected spikes per sampling bin unless explicitly converted to rate. [S15] |
| OASIS/MLspike/other inference outputs | In-memory or exported numeric arrays with a schema | Inferred events, weights, or rates and method parameters | Generic mapped interface initially; advertise dedicated support only after verifying a concrete output schema. |

Suite2p documents `spks.npy` as deconvolved traces. It must not automatically become integer spike counts. Its current `iscell.npy` may also be more recent than a prior MATLAB export; the reader should state which curation source was used. [S8]

CaImAn's denoised calcium and deconvolved events remain separate streams. CASCADE predictions require their own signal interpretation. A universal field named `spikes` would obscure these distinctions. [S9, S15]

## 5. Electrophysiology: sorted results and acquisition data

Neuropixels is probe hardware. A complete workflow can combine SpikeGLX or Open Ephys acquisition, Kilosort sorting, Phy curation, and NWB export. Each layer needs its own interface.

### Sorted activity

| Workflow | Input or adapter | MazePy responsibility | Priority |
|---|---|---|---|
| Kilosort 1/2/2.5/3/4 outputs | Kilosort/Phy output directories | Read times and final unit assignments, convert sample indices with the correct rate and clock, retain metadata | First; versioned fixtures, including KS4 |
| Phy-curated results | `spike_times.npy`, final `spike_clusters.npy`, sampling metadata, cluster tables | Respect split/merged units; retain labels; make quality filtering explicit | First |
| SpikeInterface sortings | `BaseSorting`, optionally `SortingAnalyzer` and its associated recording | Preserve IDs, segment indices, timing, properties, and available analyzer results | First; broad reusable bridge |
| MountainSort 4/5, SpyKING CIRCUS 1/2, Tridesclous 1/2, HerdingSpikes | SpikeInterface sorting objects; verified output readers where available | Common bridge, then a tested compatibility matrix for exact source versions | Second; some modern methods primarily use returned objects |
| Klusta, Wave_Clus, YASS, HDsort, Combinato, MClust, and other legacy outputs | Relevant upstream reader or converted sorting object | Preserve event and unit semantics; validate each format independently | Extended backend coverage |
| Neurosuite/NeuroScope | Associated `.res`, `.clu`, and session metadata | Preserve shank/unit identity and sample-to-time mapping | Second, through a verified backend |
| CellExplorer | `*.spikes.cellinfo.mat`, optional `*.cell_metrics.cellinfo.mat` and session metadata | Spikes plus per-unit metadata; avoid assuming metadata and spike tables share row order | Second [S18] |
| IBL/ALF outputs | ALF arrays or upstream sorting object | Probe/session namespaces, original cluster IDs, times, and quality annotations | Extended backend coverage |
| Vendor-provided sorted spikes | Supported Blackrock, Neuralynx, Plexon, or other upstream sorting readers | Common event representation with source properties | Extended backend coverage |

Sources for backend availability and object-based sorter workflows: [S2, S3]. Kilosort documents exported spike times as sample indices and provides separate cluster and template information. [S21] The importer should use final cluster assignments when present, record the source file selected, and preserve original IDs even when an internal contiguous index is used.

The observation duration cannot generally be inferred from the last spike. Obtain it from the acquisition metadata or recording intervals, or require the caller to supply it. Silent periods are part of the scientific observation time.

Loading already sorted output must not require installing or running its sorting algorithm. Optional future sorting execution should use the upstream pipeline, with its own dependencies and settings. Import, quality filtering, and sorting execution are separate features.

### Raw recordings and signals

| Acquisition family | Typical source | Proposed route |
|---|---|---|
| Neuropixels/SpikeGLX | `.bin` plus `.meta`, distinct AP/LF or other streams | SpikeInterface recording adapter; compressed IBL variants only when the relevant backend is installed and tested |
| Open Ephys | Recording directory, binary/metadata/timestamp files; legacy layouts | SpikeInterface/Neo adapter, with layout-specific tests |
| Generic extracellular binary | `.dat`/`.bin` plus supplied dtype, scaling, channel order, rate, and segment metadata | Explicit binary schema; no inference from extension alone |
| Intan | `.rhd`/`.rhs` and documented split layouts | Backend adapter |
| Blackrock | `.ns*` continuous streams and `.nev` events/spikes | Backend adapter |
| Neuralynx | `.ncs` and associated event/spike files | Backend adapter; retain discontinuities |
| Plexon | `.plx`/`.pl2` | Backend adapter, subject to format dependencies |
| TDT | Block directory and accompanying stream/event files | Backend adapter |
| SpikeGadgets | `.rec` and associated metadata | Backend adapter |
| MEA systems | Maxwell, MCS, and Biocam vendor formats | Reader-specific integration; distinguish analog channels and sorted units |
| Additional laboratory systems | CED/Spike2, Axona, AlphaOmega, White Matter, and others | Advertise only readers verified through the chosen backend |

These are targets enabled by the backend strategy, not claims of current MazePy compatibility. Consult the upstream extractor and IO lists for exact supported variants. [S2, S4]

## 6. Additional modalities and shared file formats

| Modality/workflow | Target interface | Scope |
|---|---|---|
| Voltage imaging | VolPy estimates/dictionaries; explicit array mappings for other pipelines | Import optical traces, subthreshold traces, detection indices, footprints, and sample timing. Keep optical units distinct from calibrated voltage. [S14] |
| Photometry | Doric `.doric`/CSV, pyPhotometry `.ppd`, TDT, processed arrays; optional NeuroConv-to-NWB route | Retain signal/reference roles, region/fiber IDs, native clocks, and digital events. [S19, S20, S11] |
| Intracellular recordings | Neo-backed ABF and supported other readers; NWB patch-clamp series | Preserve sweeps, stimulus/current/voltage channels, and clamp mode. [S4, S6] |
| EEG/ECoG/MEG | MNE objects and supported FIF, EDF/BDF, BrainVision, and EEGLAB routes | Retain channels, units, annotations, reference/montage, and epochs. [S17] |
| Cross-session cell/unit identity | Explicit identity-mapping tables linked to recording IDs | Import the correspondence and confidence separately from activity; never equate matching integer IDs across sessions automatically |

| General input/output | Planned behavior |
|---|---|
| NumPy arrays and SciPy sparse matrices | Direct construction; explicit axis orientation and signal interpretation; retain sparse event/count data where useful |
| `.npy` / `.npz` | First-release small-data route; arrays plus a versioned manifest or documented required keys |
| CSV / TSV | Long-form events or explicitly mapped trace tables; named time and ID columns |
| MATLAB `.mat` | Support conventional MAT files and v7.3/HDF5 through appropriate readers; require key mapping for arbitrary structs/cell arrays |
| HDF5 | Read known pipeline schemas or user-selected dataset paths; `.h5` alone does not define a scientific schema |
| Zarr | Later native large-data storage and lazy pipeline inputs; versioned layout and tested dependency versions |
| xarray / NetCDF | Named dimensions and coordinates; preserve meaning when converting |
| NWB | First-release read support for selected `Units`, electrical signals, ROI responses, intervals, and behavior; add export after round-trip tests |
| Legacy Python pickle | Explicit compatibility import for known/trusted schemas; portable exports should use documented arrays or scientific containers |
| Pynapple / Neo / SpikeInterface objects | Optional in-memory adapters; preserve support intervals, units, IDs, and metadata |

NWB is the principal standardized exchange target. It is not a requirement to convert every file before analysis. Raw HDF5, NWB, and proprietary HDF5 share a storage technology but have different schemas. NWB itself can contain several candidate streams; import should expose stream selection. [S6, S7]

## 7. Internal data model and adapter contract

Keep the agreed top-level architecture. Data containers belong in `data/`, numerical analysis in `neural/`, and readers/writers in `io/`.

| Component | Proposed contents |
|---|---|
| `data/time.py` | Regular/irregular time axes, observation intervals, source clock IDs |
| `data/neural.py` | `NeuralRecording`, `SampledActivity`, `SpikeEvents`, `BinnedActivity`, thin modality specializations |
| `data/session.py` | Multiple named streams, tables, events, covariates, clock mappings, and session metadata |
| `data/results.py` | Neural trajectories, tuning curves, processing records, and masks |
| `neural/` | Alignment, preprocessing, temporal aggregation, generalized tuning, statistics, and numerical kernels |
| `io/neural/` | Adapter registry; generic, Suite2p, CaImAn/CNMF-E, Phy/Kilosort, SpikeInterface, NWB, and later readers |
| `tests/io/neural/` | Reader fixtures, source-version tests, metadata checks, and cross-format equivalence checks |

A session can contain calcium F, deconvolved activity, ephys AP, LFP, behavior, and TTL events at different rates. Keep those streams distinct and synchronize them through explicit mappings. Do not automatically resample everything to one large matrix.

Every reader should implement:

1. `probe(source)`: identify candidate format, available streams, required parameters, and confidence without loading all samples.
2. `read(source, ...)`: return a session/recording and a structured import report.
3. `capabilities()`: state what can be loaded, sliced lazily, and preserved.

Required import-report information includes source paths and schema/version, axis mapping, sample rate or timestamps, clock, signal meaning, units/scaling, original IDs, selections, missing metadata, and any assumptions explicitly supplied by the caller. Strong format detection should inspect signatures and required fields; ambiguous files should expose alternatives.

Preserve ROI masks, electrode positions, probe identity, quality labels, and available waveform/template metadata in linked tables. Large movies, waveforms, and raw recordings should be referenced or opened lazily. Import must not calculate expensive waveforms, quality metrics, baselines, or deconvolution implicitly.

For large time-major source arrays, expose logical N-by-T access without eagerly transposing/copying the complete recording. Lazy readers must manage backing file lifetime explicitly, for example through a context manager and `close()`.

Keep physical storage units separate from calibrated units, retain gain/offset metadata, and record whether timestamps were measured or reconstructed from a sampling rate. Missing rate/clock data must be resolved before physical rates or behavior alignment are computed.

## 8. Proposed Python interface

These names illustrate the intended API; they are not available in the current package.

```python
from mazepy import io, neural

# Inspect before choosing a representation.
info = io.inspect_neural("suite2p/")
session = io.read_suite2p("suite2p/", timestamps=frame_times)
activity = session.signals["plane0.deconvolved"]

# Curated electrophysiology with a known observation interval.
session = io.read_phy("phy/", observation_intervals=[(0.0, 1800.0)])
spikes = session.signals["spikes"]

# Any compatible sorting object can enter through the same bridge.
spikes = io.from_spikeinterface_sorting(
    sorting, recording=recording, clock="ephys"
)

# The statistical engine accepts the shared representation.
trajectory = neural.to_neural_trajectory(spikes, bin_edges=time_edges)
tuning = neural.tuning_curve(
    activity, covariate=position, bins=spatial_bins
)
```

Convenient object methods may delegate to these numerical functions. Source-specific keywords should be handled by readers; tuning functions should not need to know whether an event came from Kilosort or a different sorter.

Generic imports should require `signal_kind` and `unit` whenever they cannot be established from a known schema. Quality labels should be preserved by default, with explicit selection such as good units or accepted ROIs. Any filtering changes must be recorded.

## 9. Shared processing and efficiency

First implement validation, unit/time selection, interval restriction, alignment, event counting, temporal windows, occupancy/exposure, and generalized tuning. Build spatial rate maps from those operations when spatial binning is available. Preserve the distinction between counts divided by exposure and duration-weighted means of analog signals.

Calcium preprocessing should offer explicit neuropil correction, baseline/dF/F calculation, optional normalization, and event-detection interfaces, with all parameters retained. Importing an existing result must not repeat processing automatically. Voltage/LFP filtering should expose sampling assumptions and artifact masks; downsampling needs appropriate filtering.

Neural trajectories must retain window boundaries, effective exposure, masks, units, and source identity. Tuning curves must retain bin definitions, occupancy or per-signal exposure, numerator information, and masks. Smooth event counts and occupancy separately before division. Missing observations and unvisited bins must remain distinguishable from observed zero activity.

Use NumPy/SciPy for the reference implementations and selective Numba for measured numerical bottlenecks, as discussed previously. Keep Cython an option for kernels that justify it. Preserve sparse events and chunked/lazy traces. Benchmark cold and repeated execution, peak memory, and scaling with neurons, samples, events, and variable bins. Do not allocate full raw Neuropixels recordings to compute summaries that can be accumulated in chunks.

## 10. Milestones and release gates

| Milestone | Deliverable | Completion criterion |
|---|---|---|
| A: Common model | Time axes, intervals, sampled/event/binned objects, generic array construction, core aggregation and tuning | Synthetic cases verify exposure, counts, dimensions, masks, and floating-point preservation |
| B: Primary workflows | Suite2p, CaImAn/CNMF-E, Kilosort/Phy, SpikeInterface sorting objects, selected NWB read support | Small representative source fixtures produce correct shared objects and matching analyses |
| C: Practical breadth | Lazy raw SpikeInterface access, Neo conversion, MiniAn, EXTRACT, Inscopix, CellExplorer/NeuroScope, VolPy, CASCADE | Each named reader has a documented capability/version matrix and a real or official fixture |
| D: Exchange and scaling | NWB export for supported fields, chunked persistence, Pynapple conversion, wider backend integrations | Round-trip equivalence for supported data and bounded-memory integration checks |
| E: Additional modalities | Photometry, MNE-backed EEG/ECoG/MEG, intracellular sweeps, additional requested readers | Modality-specific metadata and valid analysis operations are tested and documented |

For Milestone B, build a thin vertical slice for each major representation before implementing a large catalogue: load traces/spikes, align to a known covariate, compute a neural trajectory and tuning curve, and save an interpretable result.

Reader acceptance must cover more than successful file loading:

- Correct source axis order, units, gain/offset, and sample-index conversion.
- Original unit/ROI IDs, curated merges/splits, explicit quality filtering, and missing metadata.
- Silent observation periods, dropped frames, recording gaps, separate segments, and multiple clocks.
- Multiple imaging planes/probes and separate streams.
- Lazy slicing and file lifetime; predictable materialization and memory use.
- Count conservation and matching results for equivalent array/event/backend/NWB inputs.
- Numerical preservation of floating deconvolution outputs and expected counts.
- Versioned fixtures and useful errors for unsupported schema variants.
- Windows paths, Unicode names, and environments without optional pipeline dependencies.

Optional dependencies should be grouped by reader/backend, such as ephys, NWB, Zarr/xarray, MNE, and vendor SDKs. Core imports should remain usable without Qt, GPU frameworks, image-extraction packages, or sorting executables. Converting saved results usually needs fewer dependencies than rerunning the original pipeline.

Implementation should begin with Milestones A and B. This gives your Miniscope/CNMF-E and Suite2p work an immediate route into the same analysis engine as sorted Neuropixels data, while establishing the adapter contract needed for broader coverage.

## Sources

- **S1:** [Pynapple overview](https://pynapple.org/) and [generalized tuning curves](https://pynapple.org/user_guide/06_tuning_curves.html).
- **S2:** [SpikeInterface extractors and supported readers](https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html); [Phy/Kilosort extractor source](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/extractors/phykilosortextractors.py).
- **S3:** [SpikeInterface sorter wrappers and returned sorting objects](https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html).
- **S4:** [Neo reading and analysis](https://neo.readthedocs.io/en/stable/read_and_analyze.html) and [implemented IO modules](https://neo.readthedocs.io/en/stable/iolist.html).
- **S5:** [Elephant overview](https://elephant.readthedocs.io/en/latest/).
- **S6:** [PyNWB extracellular electrophysiology](https://pynwb.readthedocs.io/en/latest/tutorials/domain/ecephys.html) and [NWB file basics](https://pynwb.readthedocs.io/en/latest/tutorials/general/plot_file.html).
- **S7:** [PyNWB calcium imaging and ROI response series](https://pynwb.readthedocs.io/en/stable/tutorials/domain/ophys.html).
- **S8:** [Suite2p output specification](https://suite2p.readthedocs.io/en/latest/outputs/).
- **S9:** [CaImAn getting started and result fields](https://caiman.readthedocs.io/en/latest/Getting_Started.html) and [core functions and persistence](https://caiman.readthedocs.io/en/latest/core_functions.html).
- **S10:** [MiniAn variables and xarray/Zarr persistence](https://minian.readthedocs.io/en/stable/tips/variables.html).
- **S11:** [NeuroConv conversion gallery](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/index.html).
- **S12:** [NeuroConv CNMF-E MATLAB conversion](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/segmentation/cnmfe.html).
- **S13:** [EXTRACT repository](https://github.com/schnitzer-lab/EXTRACT-public) and [NeuroConv EXTRACT MATLAB conversion](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/segmentation/extract.html).
- **S14:** [CaImAn voltage functionality](https://github.com/flatironinstitute/CaImAn) and [VolPy implementation](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/volpy/spikepursuit.py).
- **S15:** [CASCADE output interpretation and examples](https://github.com/HelmchenLabSoftware/Cascade/blob/master/README.md).
- **S16:** [pyisx supported Inscopix objects](https://inscopix.github.io/pyisx/overview.html) and [reading cell/event sets](https://inscopix.github.io/pyisx/examples/readData.html).
- **S17:** [MNE importing EEG formats](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html).
- **S18:** [CellExplorer data structure and format](https://cellexplorer.org/datastructure/data-structure-and-format/).
- **S19:** [NeuroConv Doric photometry formats](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/fiberphotometry/doric_fp.html).
- **S20:** [NeuroConv pyPhotometry interfaces](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/fiberphotometry/pyphotometry_fp.html).
- **S21:** [Kilosort exported-file specification](https://kilosort.readthedocs.io/en/stable/export_files.html).
- **S22:** [MIN1PIPE output description](https://github.com/JinghaoLu/MIN1PIPE).
- **S23:** [Miniscope announcement of the MiniAn repository transfer](https://miniscope.org/wiki/Transfer_and_Modernization_of_Minian).

## 11. Mandatory example and real-data validation

Added November 3, 2026 at the user's request. Follow [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md) and [the machine-readable registry](../benchmarks/validation_registry.json). All implemented scientific APIs must pass their applicable executable example tests. Neural/behavioral processing and import/export methods require real recorded-data tests with independent references in addition to synthetic edge cases. Maintain distinct implementation, fixture, execution, and per-method validation statuses. A synthetic suite or a successful file load never establishes real-data validation of an algorithm. Missing/skipped checks remain pending. Geometry retains exact analytical examples as its correctness oracle and tracks recorded-trajectory replay separately.

A pinned 16,046-byte CASCADE DS01 recording has been acquired for fixture-schema checks and later paired calcium/spike tests. It has no measured spatial behavior and is not a native Suite2p/CaImAn/Kilosort/NWB output. Additional source-format and behavior recordings are required for those methods. Keep test identifiers, data provenance/checksums, expected results/tolerances, held-out benchmark criteria, tested commit, environment, and observed run evidence in the registry and checklist.
