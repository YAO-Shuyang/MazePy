"""Ensure the maintained checklist inventories the exposed scientific API."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validation_registry_covers_public_api():
    registry = json.loads(
        (ROOT / "benchmarks/validation_registry.json").read_text(encoding="utf-8")
    )
    entries = registry["methods"]
    implemented = {
        item["api"]: item
        for item in entries
        if item["implementation_status"] == "implemented"
    }
    assert len(implemented) == len(entries), "Duplicate or unimplemented API row"
    exposed = set()
    for module_name in registry["api_modules"]:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isclass(obj) and not issubclass(obj, BaseException):
                for method_name, member in vars(obj).items():
                    public = (
                        not method_name.startswith("_") or method_name == "__init__"
                    )
                    if public and (callable(member) or isinstance(member, property)):
                        exposed.add(f"{module_name}.{name}.{method_name}")
            elif inspect.isfunction(obj):
                exposed.add(f"{module_name}.{name}")
    assert exposed == set(implemented), (
        f"Unregistered: {sorted(exposed - set(implemented))}; "
        f"stale: {sorted(set(implemented) - exposed)}"
    )
    for entry in entries:
        assert entry["example_test_files"], entry["api"]
        for path in entry["example_test_files"] + entry["real_data_test_files"]:
            assert (ROOT / path).is_file(), (entry["api"], path)
        if entry["required_example"] == "recorded":
            assert entry["real_data_test_files"], entry["api"]
        if entry["real_data_status"] == "passed":
            assert entry["real_data_test_files"], entry["api"]
            assert entry.get("evidence"), "A real-data PASS needs observed run evidence"
