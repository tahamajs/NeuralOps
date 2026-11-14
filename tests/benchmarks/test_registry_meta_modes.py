# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying canonical coverage for all 12 operational benchmark modes."""

from __future__ import annotations

import pytest

from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkRegistry,
    BenchmarkSpec,
    OPERATIONAL_BENCHMARK_MODES,
    ensure_registry_populated,
    get_benchmark,
    get_registry,
    list_benchmarks,
    register_benchmark,
)


def test_twelve_operational_modes_defined() -> None:
    """Verify that exactly 12 canonical operational modes are defined."""
    assert len(OPERATIONAL_BENCHMARK_MODES) == 12
    modes = BenchmarkRegistry.get_operational_modes()
    assert len(modes) == 12
    assert modes == OPERATIONAL_BENCHMARK_MODES


def test_all_twelve_modes_have_benchmarks() -> None:
    """Assert all 12 operational modes have at least one registered benchmark."""
    ensure_registry_populated()
    registry = get_registry()

    for mode in OPERATIONAL_BENCHMARK_MODES:
        # Check list_benchmarks returns at least one benchmark for the mode
        benchmarks = list_benchmarks(mode=mode)
        assert len(benchmarks) >= 1, f"Mode '{mode}' has no registered benchmarks"

        # Check __meta__:<mode> entry is registered
        meta_key = f"__meta__:{mode}"
        assert meta_key in registry, f"Missing meta benchmark key '{meta_key}'"
        spec = registry[meta_key]
        assert isinstance(spec, BenchmarkSpec)
        assert spec.category == "meta"
        assert spec.target_name == mode

        # Check get_benchmark finds the meta benchmark and direct alias
        assert get_benchmark(meta_key) is not None
        assert get_benchmark(mode) is not None

        # Verify validation check
        assert BenchmarkRegistry.is_valid_mode(mode) is True


def test_invalid_mode_handling() -> None:
    """Verify invalid operational modes are properly rejected."""
    assert BenchmarkRegistry.is_valid_mode("nonexistent-mode") is False

    with pytest.raises(ValueError, match="Invalid benchmark mode"):
        @register_benchmark(name="invalid_mode_test", mode="nonexistent-mode")
        def _dummy() -> None:
            pass
