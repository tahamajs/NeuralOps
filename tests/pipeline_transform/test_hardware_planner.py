#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Hardware-Aware Resource Planner (Sub-Agent D2).
"""

import pytest

from neuralops.pipeline_transform.hardware_planner import (
    KNOWN_HARDWARE_PROFILES,
    HardwareProfile,
    ResourceEstimate,
    estimate_resources,
)
from neuralops.recipe.schema import Recipe


def test_d2_estimate_resources_returns_plausible_numbers():
    """Verify estimate_resources produces realistic VRAM, latency, parameter deltas."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "resource-test-recipe",
            "version": "1.0.0",
            "description": "Test recipe for resource estimation",
        },
        "model": {"source": "llama-3", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    est = estimate_resources(recipe, hardware_profile="rtx4090", base_parameters=125_000_000)

    assert isinstance(est, ResourceEstimate)
    assert est.peak_vram_mb > 0.0
    # 125M params in FP16 is ~250MB, with overhead peak VRAM is ~300-500MB
    assert 200.0 < est.peak_vram_mb < 2000.0
    assert est.total_time_seconds > 0.0
    assert est.parameters_after < est.parameters_before
    assert est.parameter_delta_pct < 0.0
    assert est.confidence == "high"
    assert est.hardware_profile["name"] == "NVIDIA GeForce RTX 4090"


def test_d2_warns_on_cpu_or_missing_device():
    """Verify warning is emitted when running on CPU without hardware profile specified."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "cpu-warn-test",
            "version": "1.0.0",
            "description": "Test CPU fallback warning",
        },
        "model": {"source": "toy", "dtype": "float32"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    # Calling without specifying hardware profile on non-CUDA machine
    import torch

    if not torch.cuda.is_available():
        est = estimate_resources(recipe)
        assert any("assume" in w.lower() for w in est.warnings)


def test_d2_known_rtx4090_profile_analytical_bounds():
    """Verify RTX 4090 profile estimates match analytical model within bounds."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "rtx4090-test",
            "version": "1.0.0",
            "description": "Test RTX 4090 scaling",
        },
        "model": {"source": "llama-3-8b", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    # 1 Billion parameter model test
    est = estimate_resources(recipe, hardware_profile="rtx4090", base_parameters=1_000_000_000)

    # 1B params in FP16 = ~2000 MB weights, peak VRAM around 3000-5000 MB
    assert 2500.0 < est.peak_vram_mb < 6000.0
    # Transformation time for 1B params on RTX 4090 should be roughly 20-60s
    assert 10.0 < est.total_time_seconds < 120.0


def test_d2_activation_memory_and_separate_vram_fields():
    """Verify weights_only_vram_mb and estimated_peak_vram_mb are returned separately with activation memory."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "act-test",
            "version": "1.0.0",
            "description": "Test activation memory calculation",
        },
        "model": {"source": "llama-3-8b", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    est1 = estimate_resources(
        recipe,
        hardware_profile="rtx4090",
        base_parameters=100_000_000,
        batch_size=1,
        seq_len=1024,
    )
    est2 = estimate_resources(
        recipe,
        hardware_profile="rtx4090",
        base_parameters=100_000_000,
        batch_size=4,
        seq_len=2048,
    )

    # Weights only VRAM should be identical between both runs (~190.7 MB)
    assert est1.weights_only_vram_mb == est2.weights_only_vram_mb
    assert est1.weights_only_vram_mb > 0.0

    # Peak VRAM must be greater than weights-only VRAM due to activations and scratch space
    assert est1.estimated_peak_vram_mb > est1.weights_only_vram_mb
    assert est2.estimated_peak_vram_mb > est1.estimated_peak_vram_mb

    d1 = est1.to_dict()
    assert "weights_only_vram_mb" in d1
    assert "estimated_peak_vram_mb" in d1
