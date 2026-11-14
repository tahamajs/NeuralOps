import pytest

from neuralops.benchmarks.multi_seed import (
    CANONICAL_7_SEEDS,
    DUAL_TIER_SEEDS,
    ENTERPRISE_5_SEEDS,
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
