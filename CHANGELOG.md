# Changelog

## Unreleased

- Add the `mazepy.data` shared recording interface and sampled, calcium, sparse
  spike-event, binned-activity, timestamp, and observation-interval containers.
- Preserve signal IDs, units, clocks, per-signal masks, floating amplitudes,
  silence, and immutable source data through explicit selection operations.
- Require explicit spike observation time and reject ambiguous partial-bin slices.
- Add per-method recorded-data checks, a runnable CASCADE example, and source
  fingerprints in the maintained validation ledger.
- Include the data package in wheels and verify it on Linux and Windows CI.

## 0.1.0.dev0

- Establish the agreed repository architecture with empty future modules.
- Implement immutable polygonal arenas, internal walls, and solid obstacles.
- Validate complete walls against concave boundaries and obstacles.
- Add whole-segment and polyline collision checks with explicit contact rules.
- Preserve local connectivity at wall junctions and obstacle point contacts.
- Add lazy corner visibility graphs, continuous shortest paths, pairwise batch
  distances, and bounded endpoint-visibility caches.
- Add corner-preserving path sampling and geometry fingerprints.
- Add analytic, numerical-invariance, and independent grid-connectivity tests.

This development scaffold does not include the previous experiment pipelines,
binning, neural analysis, simulation, RL, plotting, or a graphical editor.
