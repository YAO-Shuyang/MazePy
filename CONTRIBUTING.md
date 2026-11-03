# Contributing

Install development dependencies with `python -m pip install -e ".[dev]"`.
Before submitting a change, run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Implementation currently lives in `src/mazepy/geometry/` and `src/mazepy/data/`.
Future module folders contain only `.gitkeep` until explicitly started.
Use PEP 8 style, descriptive names, type annotations, and docstrings for public
behavior. Geometric calculations use explicit physical units and `(x, y)` pairs.

Preserve geometry immutability and explicit contact rules. Changes to numerical
tolerance or wall-junction behavior need a concrete geometric regression case.
Retain tests that compare reachability with independent grid flood fill and
shortest distances with analytic examples. Avoid introducing dependencies on
simulation, neural analysis, plotting, or GUI tools into geometry.

To build distributions, run `python -m build`. The repository source archive
retains the future-module scaffold; the wheel installs geometry and signal data.

Data containers must stay independent of analysis, IO, Qt, and GPU frameworks.
Preserve N-by-T orientation, signal identities, clock labels, units, valid masks,
and explicit observation support. Do not silently round inferred activity, remove
missing frames, guess a spike recording's duration, or truncate a count bin.


## Scientific example validation

Follow [the validation checklist](docs/VALIDATION_CHECKLIST.md) and keep the
[machine-readable registry](benchmarks/validation_registry.json) current.
Every implemented scientific method needs relevant executable examples;
neural/behavior processing and IO methods additionally need published recorded
data and an independent reference. File-loading checks, synthetic recordings,
and skipped tests do not count as method-specific real-data validation.

The development installation includes SciPy for the small published MAT fixture.
Run its checks with `python -m pytest tests/real_data -q -s`; the full test command
above includes these checks. Preserve and cite the exact source data and keep
real-data status pending until the corresponding method has actually passed.
