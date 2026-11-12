#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test suite verifying all 17 Group 3 Compressors (Keys 35-51).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor

GROUP3_KEYS = [
    "fwsvd_compressor",
    "gfws",
    "gptq",
    "grecodec_compressor",
    "hadamard_incoherent_svd",
    "hashed_net_compressor",
    "huffman_coder",
    "hybrid_svd_layer",
    "hybridkv_compressor",
    "kmeans_quantize",
    "kronecker_compressor",
    "kronecker_tt_lora",
    "kvtc_compressor",
    "liftr",
    "lillama_compressor",
    "lsc_compressor",
    "mamba_shedder",
]


class BenchmarkMLP(nn.Module):
    def __init__(self, in_features: int = 256, hidden: int = 256, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _get_compressor_instance(key: str, model: nn.Module, rank: int = 32):
    cls = COMPRESSOR_REGISTRY.get(key)
    kwargs = {"rank": rank, "fallback_to_original": False, "max_error": 1.0, "target_attention_only": False}
    try:
        return create_compressor(key, model=model, **kwargs)
    except Exception:
        try:
            return cls(model=model, **kwargs)
        except Exception:
            try:
                return cls(model, **kwargs)
            except Exception:
                try:
                    return cls(model=model)
                except Exception:
                    return cls(model)


@pytest.mark.parametrize("key", GROUP3_KEYS)
def test_group3_registration(key: str):
    """Verify that every compressor in Group 3 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP3_KEYS)
def test_group3_mlp_compression_and_forward(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction on MLP."""
    model = BenchmarkMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = _get_compressor_instance(key, model, rank=32)
    if hasattr(compressor, "compress") and callable(compressor.compress):
        res = compressor.compress()
        compressed = res[0] if isinstance(res, tuple) else res
    else:
        compressed = compressor

    # Forward pass check with gradient tracking
    x = torch.randn(2, 256, requires_grad=True)
    out = compressed(x)
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    elif isinstance(out, tuple):
        out = out[0]

    assert out.shape == (2, 128), f"Wrong output shape for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs encountered in {key}"

    # Autograd backward gate: verify gradient propagation
    loss = out.sum()
    loss.backward()
    trainable_params = [p for p in compressed.parameters() if p.requires_grad]
    if trainable_params:
        grads = [p.grad for p in trainable_params if p.grad is not None]
        assert len(grads) > 0, f"No parameter gradients computed during backward pass for {key}"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradient detected in {key} parameters"
    assert x.grad is not None, f"Input gradient missing during backward pass for {key}"
    assert torch.isfinite(x.grad).all(), f"Non-finite input gradient detected in {key}"

    # Physical reduction check
    comp_params = sum(p.numel() for p in compressed.parameters())
    param_ratio = orig_params / max(comp_params, 1)

    # Check sparsity
    total_zeros = sum((p == 0).sum().item() for p in compressed.parameters())
    total_elem = sum(p.numel() for p in compressed.parameters())
    sparsity = total_zeros / max(1, total_elem)
    sparsity_ratio = 1.0 / max(1e-9, 1.0 - sparsity)

    # Check bit reduction
    bit_reduction = 1.0
    if hasattr(compressor, "effective_bits"):
        bit_reduction = 32.0 / compressor.effective_bits
    elif hasattr(compressed, "effective_bits"):
        bit_reduction = 32.0 / compressed.effective_bits

    eff_ratio = max(param_ratio, sparsity_ratio, bit_reduction)
    assert eff_ratio > 1.05, f"Compressor {key} failed reduction: effective_ratio={eff_ratio:.2f}x <= 1.05x"


@pytest.mark.parametrize("key", GROUP3_KEYS)
def test_group3_bert_compression_and_forward(key: str):
    """Verify compression and forward pass on a 1-layer BERT model."""
    config = BertConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=512,
        vocab_size=500,
    )
    model = BertModel(config)
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = _get_compressor_instance(key, model, rank=32)
    if hasattr(compressor, "compress") and callable(compressor.compress):
        res = compressor.compress()
        compressed = res[0] if isinstance(res, tuple) else res
    else:
        compressed = compressor

    input_ids = torch.randint(0, 500, (2, 16))
    out = compressed(input_ids)
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    elif isinstance(out, tuple):
        out = out[0]

    assert out.shape == (2, 16, 256), f"Wrong output shape for BERT with {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs in BERT with {key}"


GROUP3_RANK_KEYS = [
    "fwsvd_compressor",
    "gfws",
    "grecodec_compressor",
    "hadamard_incoherent_svd",
    "kronecker_compressor",
    "lillama_compressor",
]


@pytest.mark.parametrize("key", GROUP3_RANK_KEYS)
def test_group3_monotonic_pareto(key: str):
    """Verify that decreasing ranks [128, 64, 32, 16] yield strictly decreasing parameter counts and increasing reduction ratios."""
    ranks = [64, 32, 16, 8]
    prev_params = float("inf")
    prev_ratio = 0.0

    for rank in ranks:
        model = BenchmarkMLP()
        orig_params = sum(p.numel() for p in model.parameters())

        compressor = _get_compressor_instance(key, model, rank=rank)
        if hasattr(compressor, "compress") and callable(compressor.compress):
            res = compressor.compress()
            comp_model = res[0] if isinstance(res, tuple) else res
        else:
            comp_model = compressor

        assert comp_model is not None, f"Compressor {key} returned None at rank {rank}"

        curr_params = sum(p.numel() for p in comp_model.parameters())
        curr_ratio = orig_params / max(1, curr_params)

        assert curr_params < prev_params, (
            f"Pareto rank monotonicity violated for {key} at rank {rank}: "
            f"current params ({curr_params}) >= previous params ({prev_params})"
        )
        assert curr_ratio > prev_ratio, (
            f"Pareto reduction ratio monotonicity violated for {key} at rank {rank}: "
            f"current ratio ({curr_ratio:.3f}) <= previous ratio ({prev_ratio:.3f})"
        )
        assert curr_ratio > 1.05, (
            f"Compressor {key} at rank {rank} failed physical reduction contract: {curr_ratio:.3f} <= 1.05"
        )

        # Autograd backward gate across each Pareto rank
        x = torch.randn(2, 256, requires_grad=True)
        out = comp_model(x)
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 128), f"Wrong output shape at rank {rank} for {key}"
        assert torch.isfinite(out).all(), f"Non-finite output at rank {rank} for {key}"
        loss = out.sum()
        loss.backward()
        trainable = [p for p in comp_model.parameters() if p.requires_grad]
        if trainable:
            grads = [p.grad for p in trainable if p.grad is not None]
            assert len(grads) > 0, f"No gradients at rank {rank} for {key}"
            assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradients at rank {rank} for {key}"
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"Invalid input grad at rank {rank} for {key}"

        prev_params = curr_params
        prev_ratio = curr_ratio

