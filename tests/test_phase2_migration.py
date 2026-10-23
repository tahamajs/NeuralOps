#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite verifying Phase 2 physical file migration and tier architecture (P002-P005).

Covers:
- test_layers_populated: len(Path("neuralops/layers").rglob("*.py")) >= 100
- test_ops_cuda_populated: len(Path("neuralops/ops/cuda").iterdir()) >= 100
- test_ops_triton_populated: len(Path("neuralops/ops/triton").rglob("*.py")) >= 25
- test_compression_populated: len(Path("neuralops/compression").rglob("*.py")) >= 70
- test_registry_identity: from neuralops.attentions import ATTENTION_REGISTRY; from neuralops.layers import ATTENTION_REGISTRY as NEW; assert ATTENTION_REGISTRY is NEW
- test_new_layers_path_importable: importing attention modules from neuralops.layers succeeds
- test_backward_shims_functional: importing from old paths resolves to new modules
"""

from __future__ import annotations

from pathlib import Path


def test_layers_populated():
    """Verify that neuralops/layers contains real migrated attention modules."""
    repo_root = Path(__file__).resolve().parent.parent
    layers_dir = repo_root / "neuralops" / "layers"
    py_files = [p for p in layers_dir.rglob("*.py") if p.name != "__init__.py"]
    assert len(py_files) >= 100, f"Expected >= 100 layer files, found {len(py_files)}"


def test_ops_compiled_c_populated():
    """Verify that neuralops/ops/cuda contains the native CUDA kernel packages."""
    repo_root = Path(__file__).resolve().parent.parent
    cuda_dir = repo_root / "neuralops" / "ops" / "cuda"
    items = [p for p in cuda_dir.iterdir() if p.name != "__pycache__"]
    assert len(items) >= 100, f"Expected >= 100 cuda packages/items, found {len(items)}"


def test_ops_jit_fusion_populated():
    """Verify that neuralops/ops/triton contains Triton kernel modules."""
    repo_root = Path(__file__).resolve().parent.parent
    triton_dir = repo_root / "neuralops" / "ops" / "triton"
    py_files = [p for p in triton_dir.rglob("*.py") if p.name != "__init__.py"]
    assert len(py_files) >= 25, f"Expected >= 25 triton kernel files, found {len(py_files)}"


def test_compression_populated():
    """Verify that neuralops/compression contains compressor modules."""
    repo_root = Path(__file__).resolve().parent.parent
    comp_dir = repo_root / "neuralops" / "compression"
    py_files = [p for p in comp_dir.rglob("*.py") if p.name != "__init__.py"]
    assert len(py_files) >= 70, f"Expected >= 70 compression files, found {len(py_files)}"


def test_registry_identity():
    """Verify that ATTENTION_REGISTRY in attentions and layers refer to the exact same object."""
    from neuralops.attentions import ATTENTION_REGISTRY as OLD_REG
    from neuralops.layers import ATTENTION_REGISTRY as NEW_REG

    assert OLD_REG is NEW_REG, "ATTENTION_REGISTRY must be the exact same object across tiers"


def test_new_layers_path_importable():
    """Verify that modules in neuralops.layers can be directly imported."""
    from neuralops.layers.linear.cosformer import CosformerAttention
    from neuralops.layers.hybrid.spectron_fp8_attention import SpectronFP8Attention

    assert CosformerAttention is not None
    assert SpectronFP8Attention is not None


def test_backward_shims_functional():
    """Verify that importing from legacy paths functions via shims."""
    from neuralops.attentions.linear.cosformer import CosformerAttention as OldCosformer
    from neuralops.layers.linear.cosformer import CosformerAttention as NewCosformer

    assert OldCosformer is NewCosformer
