<<<<<<< HEAD
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for MultiSeedConfig preset behavior and user seed preservation."""

from __future__ import annotations
=======
import pytest
>>>>>>> fix/benchmarks-remediation

from neuralops.benchmarks.multi_seed import (
    CANONICAL_7_SEEDS,
    DUAL_TIER_SEEDS,
    ENTERPRISE_5_SEEDS,
<<<<<<< HEAD
    SEED_PRESETS,
    TIER1_CANONICAL_SEEDS,
    TIER2_ADVERSARIAL_SEEDS,
    MultiSeedConfig,
    run_multiseed_benchmark,
)


def test_default_seeds():
    """Verify default seeds when neither seeds nor preset is specified."""
    config = MultiSeedConfig()
    assert config.seeds == list(TIER1_CANONICAL_SEEDS)


def test_preset_applied_when_no_user_seeds():
    """Preset is applied when user did not pass custom seeds."""
    for preset_name, expected_seeds in SEED_PRESETS.items():
        config = MultiSeedConfig(preset=preset_name)
        assert config.seeds == list(expected_seeds), (
            f"Preset '{preset_name}' should populate seeds with {expected_seeds}"
        )


def test_user_seeds_preserved_with_preset():
    """User-provided seeds must NOT be overridden by preset."""
    custom_seeds = [10, 20, 30]
    config = MultiSeedConfig(seeds=custom_seeds, preset="canonical7")
    assert config.seeds == custom_seeds, f"Custom seeds {custom_seeds} were overridden by preset: got {config.seeds}"

    custom_single = [9999]
    config_single = MultiSeedConfig(seeds=custom_single, preset="dual_tier")
    assert config_single.seeds == custom_single

    custom_arbitrary = [7, 13, 23, 41]
    config_arb = MultiSeedConfig(seeds=custom_arbitrary, preset="enterprise5")
    assert config_arb.seeds == custom_arbitrary


def test_run_multiseed_benchmark_preserves_custom_seeds():
    """Verify convenience functional runner preserves user-provided seeds even with preset."""
    custom_seeds = [11, 22, 33]
    result = run_multiseed_benchmark(
        modules=["attention:full_attention"],
        seeds=custom_seeds,
        preset="canonical7",
        dry_run=True,
    )
    assert result.config.seeds == custom_seeds
=======
    TIER1_CANONICAL_SEEDS,
    TIER2_ADVERSARIAL_SEEDS,
    TIER2_EXTENDED_SEEDS,
    MultiSeedConfig,
)


def test_multi_seed_preset_override():
    # When preset is given with no explicit seeds, preset applies
    cfg = MultiSeedConfig(preset="tier2")
    assert cfg.seeds == list(TIER2_EXTENDED_SEEDS)

    # When explicit custom seeds are provided alongside preset, custom seeds are preserved
    custom = [101, 202, 303]
    cfg_custom = MultiSeedConfig(seeds=custom, preset="tier2")
    assert cfg_custom.seeds == custom


def test_multi_seed_default_seeds():
    # Without preset and without explicit seeds, defaults to TIER1_CANONICAL_SEEDS
    cfg = MultiSeedConfig()
    assert cfg.seeds == list(TIER1_CANONICAL_SEEDS)


def test_multi_seed_explicit_canonical_preserved_with_preset():
    # If explicit seeds are provided even if they match TIER1_CANONICAL_SEEDS, they must not be overwritten
    explicit = list(TIER1_CANONICAL_SEEDS)
    cfg = MultiSeedConfig(seeds=explicit, preset="enterprise5")
    assert cfg.seeds == explicit


def test_multi_seed_various_presets():
    assert MultiSeedConfig(preset="canonical3").seeds == list(TIER1_CANONICAL_SEEDS)
    assert MultiSeedConfig(preset="tier1").seeds == list(TIER1_CANONICAL_SEEDS)
    assert MultiSeedConfig(preset="adversarial3").seeds == list(TIER2_ADVERSARIAL_SEEDS)
    assert MultiSeedConfig(preset="dual_tier").seeds == list(DUAL_TIER_SEEDS)
    assert MultiSeedConfig(preset="enterprise5").seeds == list(ENTERPRISE_5_SEEDS)
    assert MultiSeedConfig(preset="canonical7").seeds == list(CANONICAL_7_SEEDS)
>>>>>>> fix/benchmarks-remediation
