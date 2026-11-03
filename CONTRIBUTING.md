# Contributing

Install development dependencies with `python -m pip install -e ".[dev]"`.
Before submitting a change, run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Keep all implementation in `src/mazepy/geometry/` until another module is
explicitly started. Future module folders currently contain only `.gitkeep`.
Use PEP 8 style, descriptive names, type annotations, and docstrings for public
behavior. Geometric calculations use explicit physical units and `(x, y)` pairs.

Preserve geometry immutability and explicit contact rules. Changes to numerical
tolerance or wall-junction behavior need a concrete geometric regression case.
Retain tests that compare reachability with independent grid flood fill and
shortest distances with analytic examples. Avoid introducing dependencies on
simulation, neural analysis, plotting, or GUI tools into geometry.

To build distributions, run `python -m build`. The repository source archive
retains the empty future-module scaffold; the wheel installs the geometry code.


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
