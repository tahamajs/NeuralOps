"""Tests verifying that all quarantined legacy benchmark scripts issue DeprecationWarning.

Verifies deprecation warnings for:
- neuralops.benchmarks._legacy package and all quarantined scripts
- neuralops.benchmarks.legacy_scripts forwarder shims
- neuralops.benchmarks.legacy_root forwarder shims
- neuralops.benchmarks.cli.main() entrypoint shim
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import warnings
import pytest


def test_legacy_package_deprecation_warning() -> None:
    """Verify importing neuralops.benchmarks._legacy triggers DeprecationWarning."""
    sys.modules.pop("neuralops.benchmarks._legacy", None)
    with pytest.deprecated_call():
        importlib.import_module("neuralops.benchmarks._legacy")


def test_cli_shim_deprecation_warning() -> None:
    """Verify invoking neuralops.benchmarks.cli.main() triggers DeprecationWarning."""
    from neuralops.benchmarks.cli import main

    with pytest.deprecated_call():
        # Invoke with empty list (exits 0 with help / notice)
        main(["--help"])


def test_quarantined_legacy_scripts_deprecation_warning() -> None:
    """Verify each script in neuralops/benchmarks/_legacy issues DeprecationWarning on import."""
    repo_root = Path(__file__).resolve().parents[2]
    legacy_dir = repo_root / "neuralops" / "benchmarks" / "_legacy"
    assert legacy_dir.is_dir(), "_legacy directory must exist"

    python_scripts = sorted([p for p in legacy_dir.rglob("*.py") if p.name != "__init__.py"])
    assert len(python_scripts) >= 50, f"Expected at least 50 legacy scripts in _legacy, found {len(python_scripts)}"

    warned_count = 0
    for script_path in python_scripts:
        rel = script_path.relative_to(legacy_dir)
        mod_name = "neuralops.benchmarks._legacy." + ".".join(rel.with_suffix("").parts)

        # Clear cached module
        sys.modules.pop(mod_name, None)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                spec = importlib.util.spec_from_file_location(mod_name, script_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)
            except Exception:
                # Some scripts may fail on CUDA/MPS hardware initialization, but they
                # MUST issue DeprecationWarning prior to execution
                pass

            has_deprecation = any(issubclass(item.category, DeprecationWarning) for item in recorded)
            assert has_deprecation, f"Script {script_path.name} ({mod_name}) failed to issue DeprecationWarning"
            warned_count += 1

    assert warned_count == len(python_scripts)


def test_legacy_scripts_shims_deprecation_warning() -> None:
    """Verify shims in neuralops.benchmarks.legacy_scripts issue DeprecationWarning."""
    repo_root = Path(__file__).resolve().parents[2]
    shims_dir = repo_root / "neuralops" / "benchmarks" / "legacy_scripts"

    sample_shims = ["benchmark_config", "benchmark_logger", "gpu_master_suite"]
    for shim_name in sample_shims:
        shim_file = shims_dir / f"{shim_name}.py"
        if shim_file.exists():
            mod_name = f"neuralops.benchmarks.legacy_scripts.{shim_name}"
            sys.modules.pop(mod_name, None)
            with pytest.deprecated_call():
                importlib.import_module(mod_name)


def test_legacy_root_shims_deprecation_warning() -> None:
    """Verify shims in neuralops.benchmarks.legacy_root issue DeprecationWarning."""
    repo_root = Path(__file__).resolve().parents[2]
    shims_dir = repo_root / "neuralops" / "benchmarks" / "legacy_root"

    sample_shims = ["bench_llm_bandwidth", "benchmark_all_26_kernels", "benchmark_dynamic_range_stress"]
    for shim_name in sample_shims:
        shim_file = shims_dir / f"{shim_name}.py"
        if shim_file.exists():
            mod_name = f"neuralops.benchmarks.legacy_root.{shim_name}"
            sys.modules.pop(mod_name, None)
            with pytest.deprecated_call():
                importlib.import_module(mod_name)
