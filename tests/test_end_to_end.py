#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-End Correctness and Model Surgery Verification Suite.

Tests:
  1. test_import_neuralops(): Validates library entry points, versioning, and public symbols.
  2. test_replace_attention_distilbert(): Validates attention surgery on DistilBertModel.
  3. test_replace_attention_gpt2(): Validates attention surgery on GPT2Model.
  4. test_torch_save_load_after_surgery(): Validates torch checkpoint save/load roundtrip after surgery.
  5. test_triton_kernels_import(): Validates importability of all 26 canonical Triton kernel modules.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel, GPT2Config, GPT2Model

import neuralops
from neuralops import (
    load_model,
    replace_attention,
    compress,
    export_model,
    get_version,
)
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


CANONICAL_TRITON_KERNELS: list[str] = [
    "block_sparse",
    "cosformer_triton_kernel",
    "cur_linear",
    "differential_attention",
    "dynamic_sparse_attention",
    "elementwise",
    "flash_decoding",
    "fms_scan",
    "fused_ce",
    "fused_kd",
    "fused_norm_linear",
    "fused_rope",
    "gdn_scan",
    "gla_scan",
    "gqa",
    "hgrn2_scan",
    "hydra_attention",
    "linear_attn",
    "low_rank",
    "paged_attention",
    "retnet_retention",
    "selective_scan",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
]


def test_import_neuralops():
    """Verify neuralops library imports, exposes public APIs, versioning, and canonical registries."""
    assert hasattr(neuralops, "__version__")
    assert neuralops.__version__ == "1.4.2"
    assert get_version() == "1.4.2"

    public_callables = [load_model, replace_attention, compress, export_model]
    for fn in public_callables:
        assert callable(fn), f"Expected callable API {fn}"

    assert len(ATTENTION_REGISTRY) == 131, f"Expected 131 attention mechanisms, found {len(ATTENTION_REGISTRY)}"
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72 compression engines, found {len(COMPRESSOR_REGISTRY)}"


def test_replace_attention_distilbert():
    """Verify attention surgery on DistilBertModel with forward execution and metadata tagging."""
    config = DistilBertConfig(
        vocab_size=120,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
    )
    model = DistilBertModel(config)
    replaced_model = replace_attention(model, "cosformer")

    assert hasattr(replaced_model, "_neuralops_metadata"), "Replaced model missing _neuralops_metadata"
    meta = replaced_model._neuralops_metadata
    assert len(meta) >= 1
    assert any(entry.get("type") == "cosformer" for entry in meta)

    replaced_model.eval()
    input_ids = torch.randint(0, 120, (2, 16))
    with torch.no_grad():
        output = replaced_model(input_ids)

    assert hasattr(output, "last_hidden_state")
    assert output.last_hidden_state.shape == (2, 16, 64)
    assert not torch.isnan(output.last_hidden_state).any()
    assert not torch.isinf(output.last_hidden_state).any()


def test_replace_attention_gpt2():
    """Verify attention surgery on GPT2Model with forward execution and shape preservation."""
    config = GPT2Config(
        vocab_size=120,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2Model(config)
    replaced_model = replace_attention(model, "cosformer")

    assert hasattr(replaced_model, "_neuralops_metadata"), "Replaced model missing _neuralops_metadata"
    meta = replaced_model._neuralops_metadata
    assert len(meta) >= 1
    assert any(entry.get("type") == "cosformer" for entry in meta)

    replaced_model.eval()
    input_ids = torch.randint(0, 120, (2, 16))
    with torch.no_grad():
        output = replaced_model(input_ids)

    assert hasattr(output, "last_hidden_state")
    assert output.last_hidden_state.shape == (2, 16, 64)
    assert not torch.isnan(output.last_hidden_state).any()
    assert not torch.isinf(output.last_hidden_state).any()


def test_torch_save_load_after_surgery():
    """Verify torch save and load state_dict roundtrip preserves numerical parity after surgery."""
    config = DistilBertConfig(
        vocab_size=120,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
    )
    model1 = DistilBertModel(config)
    replaced1 = replace_attention(model1, "cosformer")
    replaced1.eval()

    input_ids = torch.randint(0, 120, (2, 16))
    with torch.no_grad():
        out1 = replaced1(input_ids).last_hidden_state

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "surgical_checkpoint.pt"
        torch.save(replaced1.state_dict(), ckpt_path)
        assert ckpt_path.is_file()

        model2 = DistilBertModel(config)
        replaced2 = replace_attention(model2, "cosformer")
        saved_weights = torch.load(ckpt_path, weights_only=True)
        replaced2.load_state_dict(saved_weights)
        replaced2.eval()

        with torch.no_grad():
            out2 = replaced2(input_ids).last_hidden_state

    assert torch.allclose(out1, out2, atol=1e-5), "Reloaded model outputs diverged from original surgically modified model"


def test_triton_kernels_import():
    """Verify that all 26 canonical Triton kernel modules import cleanly and are functional."""
    assert len(CANONICAL_TRITON_KERNELS) == 26

    imported_modules = []
    for mod_name in CANONICAL_TRITON_KERNELS:
        module_path = f"neuralops.core.triton_kernels.{mod_name}"
        mod = importlib.import_module(module_path)
        assert mod is not None, f"Failed to import {module_path}"
        imported_modules.append(mod)

    assert len(imported_modules) == 26
