# Method validation checklist

Updated: November 3, 2026. Inventory baseline: `37c876867b23595b1bfc2cc71afebe31585f5623`.

This is a maintained evidence ledger. An implementation, a working demo, a downloaded recording, and a validated scientific method are separate milestones. The machine-readable counterpart is [validation_registry.json](../benchmarks/validation_registry.json).

## Required rule

Every implemented or changed public scientific method must pass its applicable example tests (样例测试). Neural/behavioral processing and data-import methods require real recorded data in addition to synthetic edge cases. Geometry requires exact analytical examples; recorded-trajectory replay is tracked separately and must never be marked as completed by a mathematical test. Simulation requires deterministic/seeded examples and appropriate comparison to recordings before making any real-data claim.

Before a method is reported as validated, record its API and source revision, dataset/DOI, original or derived data status, source version, checksum, selected units/time interval, exact test names, independent reference or ground truth, parameters, acceptance tolerance, outcome, platform, and run evidence. A skipped test or missing fixture is pending, never a pass. A CI run must exercise the specific method; simply opening a real data file is insufficient.

For deterministic calculations use exact counts or explicit numerical tolerances against an independent implementation/reference. For statistical/inference methods declare the metric, threshold, seeds, and held-out data before tuning. Tiny fixtures provide regression coverage; broader claims need representative held-out recordings. Never derive the expected answer with the same function under test.

## Current implemented API

The source contains geometry, shared `mazepy.data` containers, and spike binning/rate normalization in `mazepy.neural`. The geometry baseline above is unchanged. Containers and spike processing have method-specific checks on the published calcium/spike fixture. Other neural algorithms and native pipeline importers remain planned. GUI/simulation archives outside this source inventory require a separate audit when integrated.

| Public API | Implementation | Analytical example tests | Real-data validation | Test location |
|---|---|---|---|---|
| `Arena2D.__init__` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.boundary` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.walls` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.obstacles` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.bounds` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.area` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.contains` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.collision_checker` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.planner` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.segment_is_valid` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Arena2D.path_is_valid` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Arena2D.shortest_path` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `Arena2D.distance` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `Arena2D.with_wall` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.without_wall` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_arena.py) |
| `CollisionChecker.__init__` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_collision.py) |
| `CollisionChecker.segment_is_valid` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_collision.py) |
| `CollisionChecker.path_is_valid` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Path.__init__` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Path.sample` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.__init__` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.graph` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.shortest_path` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.distance` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.distances` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.clear_cache` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.cache_info` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `VisibilityGraph.__init__` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `VisibilityGraph.n_nodes` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `VisibilityGraph.n_edges` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `build_visibility_graph` | Implemented | Passed (analytical suite) | Not run | [Examples](../tests/geometry/test_pathfinding.py) |

## Shared signal-container API

The abstract `NeuralRecording` interface is exercised through concrete subclasses. Inherited methods are included for each concrete public class. All 41 entries below have both structural edge cases and recorded-data tests. Validation scope is storage/selection and declared time support, rather than scientific signal inference or native pipeline compatibility. `TimeAxis.regular` is a generator checked with fixture-derived dimensions/rate; native timestamps remain unchanged.

| Public API | Example tests | Recorded-data tests | Test source |
|---|---|---|---|
| `TimeAxis.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeAxis.indices_between` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeAxis.n_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeAxis.regular` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeAxis.select_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeIntervals.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeIntervals.clip` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeIntervals.contains` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `TimeIntervals.duration` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.n_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.n_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.select_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.select_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.time_slice` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.to_numpy` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SampledActivity.with_metadata` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.n_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.n_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.select_samples` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.select_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.time_slice` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.to_numpy` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `CalciumTrace.with_metadata` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.from_unit_ids` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.n_events` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.n_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.select_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.time_slice` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `SpikeEvents.with_metadata` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.__init__` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.n_bins` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.n_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.select_bins` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.select_signals` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.time` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.time_slice` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.to_numpy` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |
| `BinnedActivity.with_metadata` | Passed | Passed (container contract) | [Case](../tests/real_data/test_signal_containers.py) |

[Local and installed-wheel evidence](../benchmarks/evidence/2026-11-03-signals-local.json) records 174 passing tests with zero skips, Python/dependency versions, exact source hashes, fixture identity, oracles, and acceptance criteria. Native source values and selected event counts match independent references. The command-line example also passes on the recorded fixture.

[Cross-platform CI evidence](../benchmarks/evidence/2026-11-03-signals-ci.json) verifies commit `65e9e531da16bc6ae346d0b85aadb97fe5bd62a0`: all 174 tests passed with zero skips on Linux Python 3.10/3.12 and Windows Python 3.12. Each job additionally installed its built wheel and passed all 87 signal/example tests. Lint and formatting passed. [Workflow run](https://github.com/YAO-Shuyang/MazePy/actions/runs/41588588818) retains dependency versions and JUnit reports. These historical results cover the stated container contracts at that revision; the current package is rechecked in the spike-processing evidence below.

## Spike processing API

| Public API | Analytical/seeded examples | Recorded-data checks | Test source |
|---|---|---|---|
| `mazepy.neural.bin_spikes` | Passed | Passed (counting and exposure) | [Cases](../tests/real_data/test_spike_binning.py) |
| `mazepy.neural.to_rate` | Passed | Passed (count/time normalization) | [Cases](../tests/real_data/test_spike_binning.py) |

[Local source and installed-wheel evidence](../benchmarks/evidence/2026-11-03-spike-processing-local.json)
records **211 passing tests, zero skips**, in each environment, plus passing lint
and formatting. The active registry covers 74 APIs: 31 geometry, 41 concrete data
APIs, and these 2 processing functions. Container checks were rerun against the
current package; the historical source fingerprints above remain unchanged.

The pinned CASCADE native AP timestamps retain 43 of 44 spikes within explicit
half-open imaging support. Counts match independently filtered `numpy.histogram`
results on uniform, irregular, and native event-edge bins. Exposure matches direct
scalar segment/bin intersections. Independently counted APs provide the rate
oracle; integrating example rates over exposure recovers exactly 43 spikes.
Counts/masks are exact; exposure tolerance is 1e-12 seconds; rates use rtol=1e-12,
atol=1e-12 Hz. A controlled recorded-event subset tests per-unit gaps without
claiming those gaps existed in the native acquisition. Multi-unit, silent-unit,
boundary, missing-exposure, and numerical-limit examples are labeled synthetic.
Source/fixture hashes, parameters, seed, environments, test nodes, and limitations
are recorded in the evidence file. Large-scale performance remains **not run**.

## Planned neural capabilities

| Capability | Implementation | Real fixture/reference still needed |
|---|---|---|
| neural alignment and timestamp conversion | Planned | cascade_ds01_theis16_cell21 |
| neural trajectories | Planned | cascade_ds01_theis16_cell21 |
| calcium preprocessing and inference | Planned | cascade_ds01_theis16_cell21; additional_held_out_calcium_records_required |
| spatial occupancy and tuning | Planned | published_spikes_and_recorded_behavior_required |
| Suite2p importer | Planned | native_suite2p_fixture_required |
| CaImAn/CNMF-E importer | Planned | native_caiman_and_matlab_fixtures_required |
| Kilosort/Phy importer | Planned | native_curated_sorting_fixture_required |
| NWB importer/exporter | Planned | published_nwb_fixture_required |

## Acquired published demo

- [CASCADE DS01 cell 21](../examples/data/cascade_ds01/README.md): **16,046 bytes**, original MAT file, one mouse V1 OGB-1 neuron, paired fluorescence and cell-attached spike timestamps. [Manifest](../examples/data/cascade_ds01/manifest.json).
- Acquisition, SHA-256 verification, and MAT/schema checks passed on Linux and Windows, Python 3.12. The complete suite passed **87 tests with zero skips** on each platform. [Run evidence](../benchmarks/evidence/2026-11-03-ci.json).
- This is recorded data; it is not a simulated MEArec recording. It contains extracted spike timestamps rather than raw ephys voltage.
- The original fixture tests establish file integrity/schema. The additional `test_signal_containers.py` cases invoke actual public container methods and establish the narrowly stated storage/selection contracts. They do not establish inference accuracy or native pipeline compatibility.
- A published recording with measured position/heading must be acquired separately before validating spatial/head-direction tuning. Native Suite2p, CaImAn/CNMF-E, Kilosort/Phy, and NWB readers each need an actual source-format fixture.

## Observed evidence

[CI run 41584441651](https://github.com/YAO-Shuyang/MazePy/actions/runs/41584441651) tested commit `199b49e5b32089641df6330ddd778c774d4c42fc` on Linux and Windows. Both passed all 87 tests and lint checks, with no skipped tests. The unchanged geometry implementation is the inventory baseline above. Fixture-schema checks observed one record with 1,164 samples, 44 native spikes, and 43 spikes within the inclusive first-to-last fluorescence sample span (96.73931126268508 seconds).

This initial evidence establishes the recorded fixture's integrity/schema and the analytical example suite. Container-contract and spike-processing evidence is recorded separately above. Planned analysis rows remain unimplemented, and geometry recorded-trajectory replay remains not run.

## Updating the ledger

1. Register each public API and identify the applicable real-data examples.
2. Add a compact, cited fixture or a reproducible fetch manifest and an independent expected result.
3. Run the full test suite and real-data tests. Preserve the run's JUnit report, commit, and environment evidence.
4. Update this checklist and the registry with the observed outcome and run link. A change to method behavior, data, or processing settings invalidates affected earlier evidence until rerun.
5. Keep missing, failed, skipped, and unimplemented states explicit in release notes. Do not promote pipeline compatibility from a generic MAT/NumPy test.

CI runs the existing analytical tests, fixture tests, and registry checks on Linux and Windows. Artifacts retain test results and the exact manifest. Repository branch-protection enforcement, if desired, is separate from the existence of a workflow.
