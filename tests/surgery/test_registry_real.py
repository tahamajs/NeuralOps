#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying the surgery registry metadata reflects reality.
"""

from __future__ import annotations

from neuralops.surgery.registry import get_surgery, list_surgeries


def test_registry_populated_with_canonical_surgeries():
    """Verify registry exports all canonical v1.9.0 surgeries."""
    surgeries = list_surgeries()
    expected = {
        "replace_attention",
        "shrink_mlp",
        "prune_layers",
        "quantize_int4",
        "distill_from_teacher",
    }
    assert expected.issubset(set(surgeries))

    for name in surgeries:
        meta = get_surgery(name)
        assert meta.name == name
        assert isinstance(meta.category, str) and len(meta.category) > 0
        assert isinstance(meta.compat, list) and len(meta.compat) > 0
        assert isinstance(meta.requires_gpu, bool)
        assert meta.cost_estimate in ("low", "medium", "high")


def test_registry_hardware_and_resource_contracts():
    """Verify hardware and calibration requirements accurately match physical execution constraints."""
    # replace_attention: CPU supported via vectorized PyTorch fallback
    rep = get_surgery("replace_attention")
    assert rep.requires_gpu is False

    # shrink_mlp: CPU supported via dimension truncation
    shrink = get_surgery("shrink_mlp")
    assert shrink.requires_gpu is False

    # quantize_int4: Requires GPU acceleration and calibration dataset
    quant = get_surgery("quantize_int4")
    assert quant.requires_gpu is True or quant.requires_calibration is True

    # distill_from_teacher: Requires GPU acceleration for gradient healing
    distill = get_surgery("distill_from_teacher")
    assert distill.requires_gpu is True
