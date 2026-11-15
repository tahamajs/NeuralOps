"""Tests verifying full benchmarks legacy sunset and v2 migration.

Validates:
1. neuralops.benchmarks.cli deprecation shim delegates properly to cli_v2.
2. neuralops.benchmarks.suites package and all suite submodules issue DeprecationWarning.
3. neuralops.benchmarks.suites_v2 contains all expected benchmark suites.
4. Legacy benchmarks modules have zero naked print statements.
5. All suite classes in suites_v2 properly subclass AbstractBaseBenchmark.
"""

from __future__ import annotations

import glob
import importlib
import re
import sys
import pytest

from neuralops.benchmarks.core.base import AbstractBaseBenchmark
import neuralops.benchmarks.suites_v2 as suites_v2


def test_cli_v2_main_delegation() -> None:
    """Verify cli.py main delegates to cli_v2.main:main with DeprecationWarning."""
    from neuralops.benchmarks.cli import main

    with pytest.deprecated_call():
        ret = main(["--help"])
        assert ret == 0


def test_suites_package_deprecation_warning() -> None:
    """Verify importing neuralops.benchmarks.suites triggers DeprecationWarning."""
    sys.modules.pop("neuralops.benchmarks.suites", None)
    with pytest.deprecated_call():
        importlib.import_module("neuralops.benchmarks.suites")


@pytest.mark.parametrize(
    "sub_mod",
    [
        "attention",
        "compression",
        "cuda_ops",
        "e2e_serving",
        "niah",
        "qwen_kd",
        "roofline",
        "triton_ops",
    ],
    ids=lambda x: f"mod_{x}",
)
def test_suites_submodules_deprecation_warning(sub_mod: str) -> None:
    submodule = sub_mod
    """Verify importing any submodule in neuralops.benchmarks.suites triggers DeprecationWarning."""
    mod_path = f"neuralops.benchmarks.suites.{submodule}"
    sys.modules.pop(mod_path, None)
    with pytest.deprecated_call():
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, "__all__") or dir(mod)


def test_suites_v2_registry_and_contracts() -> None:
    """Verify suites_v2 registry contains all canonical suites and proper class contracts."""
    expected_suites = {
        "attention",
        "compression",
        "cuda_ops",
        "e2e_serving",
        "niah",
        "qwen_kd",
        "roofline",
        "triton_ops",
    }
    suites_map = suites_v2.BENCHMARK_SUITES
    assert set(suites_map.keys()) == expected_suites
    assert set(suites_v2.list_suites()) == expected_suites

    for name in expected_suites:
        cls = suites_v2.get_benchmark_suite(name)
        assert issubclass(cls, AbstractBaseBenchmark)
        assert hasattr(cls, "setup")
        assert hasattr(cls, "run")
        assert hasattr(cls, "teardown")


def test_zero_naked_prints_in_benchmarks() -> None:
    """Verify zero naked print statements remain in benchmarks/ outside cli_v2."""
    print_patterns = re.compile(r"^\s*print\(", re.MULTILINE)
    violating_files = {}

    for py_file in glob.glob("neuralops/benchmarks/**/*.py", recursive=True):
        if "cli_v2" in py_file:
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        matches = [line.strip() for line in lines if print_patterns.match(line)]
        if matches:
            violating_files[py_file] = len(matches)

    assert not violating_files, f"Found naked print statements in: {violating_files}"
