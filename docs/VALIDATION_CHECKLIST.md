# Method validation checklist

Updated: November 3, 2026. Inventory baseline: `37c876867b23595b1bfc2cc71afebe31585f5623`.

This is a maintained evidence ledger. An implementation, a working demo, a downloaded recording, and a validated scientific method are separate milestones. The machine-readable counterpart is [validation_registry.json](../benchmarks/validation_registry.json).

## Required rule

Every implemented or changed public scientific method must pass its applicable example tests (样例测试). Neural/behavioral processing and data-import methods require real recorded data in addition to synthetic edge cases. Geometry requires exact analytical examples; recorded-trajectory replay is tracked separately and must never be marked as completed by a mathematical test. Simulation requires deterministic/seeded examples and appropriate comparison to recordings before making any real-data claim.

Before a method is reported as validated, record its API and source revision, dataset/DOI, original or derived data status, source version, checksum, selected units/time interval, exact test names, independent reference or ground truth, parameters, acceptance tolerance, outcome, platform, and run evidence. A skipped test or missing fixture is pending, never a pass. A CI run must exercise the specific method; simply opening a real data file is insufficient.

For deterministic calculations use exact counts or explicit numerical tolerances against an independent implementation/reference. For statistical/inference methods declare the metric, threshold, seeds, and held-out data before tuning. Tiny fixtures provide regression coverage; broader claims need representative held-out recordings. Never derive the expected answer with the same function under test.

## Current implemented API

The current GitHub baseline contains geometry only. No neural algorithms or importers have been implemented on that baseline. GUI/simulation archives from earlier work are outside this source inventory and need a separate audit when integrated.

| Public API | Implementation | Analytical example tests | Real-data validation | Test location |
|---|---|---|---|---|
| `Arena2D.__init__` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.boundary` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.walls` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.obstacles` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.bounds` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.area` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.contains` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.collision_checker` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.planner` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Arena2D.segment_is_valid` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Arena2D.path_is_valid` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Arena2D.shortest_path` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `Arena2D.distance` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `Arena2D.with_wall` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `Arena2D.without_wall` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_arena.py) |
| `CollisionChecker.__init__` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_collision.py) |
| `CollisionChecker.segment_is_valid` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_collision.py) |
| `CollisionChecker.path_is_valid` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_collision.py) |
| `Path.__init__` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `Path.sample` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.__init__` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.graph` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.shortest_path` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.distance` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.distances` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.clear_cache` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `PathPlanner.cache_info` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |
| `VisibilityGraph.__init__` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `VisibilityGraph.n_nodes` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `VisibilityGraph.n_edges` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_public_examples.py) |
| `build_visibility_graph` | Implemented | Pending CI | Not run | [Examples](../tests/geometry/test_pathfinding.py) |

## Planned neural capabilities

| Capability | Implementation | Real fixture/reference still needed |
|---|---|---|
| neural alignment and timestamp conversion | Planned | cascade_ds01_theis16_cell21 |
| event counting and neural trajectories | Planned | cascade_ds01_theis16_cell21 |
| calcium preprocessing and inference | Planned | cascade_ds01_theis16_cell21; additional_held_out_calcium_records_required |
| spatial occupancy and tuning | Planned | published_spikes_and_recorded_behavior_required |
| Suite2p importer | Planned | native_suite2p_fixture_required |
| CaImAn/CNMF-E importer | Planned | native_caiman_and_matlab_fixtures_required |
| Kilosort/Phy importer | Planned | native_curated_sorting_fixture_required |
| NWB importer/exporter | Planned | published_nwb_fixture_required |

## Acquired published demo

- [CASCADE DS01 cell 21](../examples/data/cascade_ds01/README.md): **16,046 bytes**, original MAT file, one mouse V1 OGB-1 neuron, paired fluorescence and cell-attached spike timestamps. [Manifest](../examples/data/cascade_ds01/manifest.json).
- Acquisition and SHA-256 verification completed. MAT/schema execution is pending CI.
- This is recorded data; it is not a simulated MEArec recording. It contains extracted spike timestamps rather than raw ephys voltage.
- Its fixture tests establish file integrity, schema, and timing assumptions only. They do not establish that a MazePy neural method is correct.
- A published recording with measured position/heading must be acquired separately before validating spatial/head-direction tuning. Native Suite2p, CaImAn/CNMF-E, Kilosort/Phy, and NWB readers each need an actual source-format fixture.

## Updating the ledger

1. Register each public API and identify the applicable real-data examples.
2. Add a compact, cited fixture or a reproducible fetch manifest and an independent expected result.
3. Run the full test suite and real-data tests. Preserve the run's JUnit report, commit, and environment evidence.
4. Update this checklist and the registry with the observed outcome and run link. A change to method behavior, data, or processing settings invalidates affected earlier evidence until rerun.
5. Keep missing, failed, skipped, and unimplemented states explicit in release notes. Do not promote pipeline compatibility from a generic MAT/NumPy test.

CI runs the existing analytical tests, fixture tests, and registry checks on Linux and Windows. Artifacts retain test results and the exact manifest. Repository branch-protection enforcement, if desired, is separate from the existence of a workflow.
