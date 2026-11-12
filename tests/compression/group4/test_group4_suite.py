#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test suite for all 16 Group 4 Compressors (Keys 52-67).
Verifies that every compressor achieves:
1. Effective compression ratio R > 1.05x.
2. 100% finite outputs (Zero NaNs, Zero Infs) on MLP and BERT architectures.
"""

import copy
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import COMPRESSOR_REGISTRY

GROUP4_KEYS = [
    "mgie_budget",
    "mlorq_compressor",
    "modegpt_compressor",
    "nmf_compressor",
    "oats_compressor",
    "optimal_subspace_projection",
    "proxsparse",
    "quarot",
    "residual_svd_compressor",
    "roast_hashed_linear",
    "rotational_slicegpt_int4",
    "rpca_compressor",
    "saes_svd_v2",
    "salr",
    "sdrm_compressor",
    "slicegpt",
]


def make_mlp():
    return nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768),
    )


def make_bert():
    cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=1,
        intermediate_size=3072,
        vocab_size=1000,
    )
    return BertModel(cfg)


def apply_compressor(key: str, model: nn.Module, rank: int = 16):
    target = COMPRESSOR_REGISTRY[key]
    kwargs = {
        "rank": rank,
        "target_rank": rank,
        "fallback_to_original": False,
        "max_error": 2.0,
    }
    instance = None
    for fn in [
        lambda: target(model, **kwargs),
        lambda: target(model, rank=rank),
        lambda: target(model),
    ]:
        try:
            instance = fn()
            break
        except Exception:
            continue

    if instance is None:
        raise RuntimeError(f"Could not instantiate compressor {key}")

    if isinstance(instance, tuple):
        comp_model = instance[0]
    elif hasattr(instance, "compress"):
        try:
            res = instance.compress()
        except TypeError:
            res = instance.compress(model)
        if isinstance(res, tuple):
            comp_model = res[0]
        else:
            comp_model = res
    else:
        comp_model = instance

    return comp_model


@pytest.mark.parametrize("key", GROUP4_KEYS)
def test_group4_mlp_compression_and_forward(key: str):
    torch.manual_seed(42)
    model = make_mlp()
    orig_params = sum(p.numel() for p in model.parameters())

    compressed = apply_compressor(key, copy.deepcopy(model), rank=16)

    # Check forward pass with gradient tracking
    x = torch.randn(2, 768, requires_grad=True)
    out = compressed(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    assert torch.isfinite(out).all(), f"{key} produced non-finite output on MLP"
    assert out.shape == (2, 768)

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

    # Check compression ratio
    comp_params = sum(p.numel() for p in compressed.parameters())
    zero_params = 0
    total_weights = 0
    for p in compressed.parameters():
        if p.dim() >= 2:
            zero_params += int((p == 0).sum().item())
            total_weights += p.numel()

    sparsity = zero_params / max(total_weights, 1)
    effective_params = comp_params * (1.0 - sparsity)
    ratio = orig_params / max(effective_params, 1)

    has_quant = any(getattr(m, "effective_bits", 32) <= 8 for m in compressed.modules())
    if has_quant:
        min_bits = min(getattr(m, "effective_bits", 32) for m in compressed.modules())
        ratio = ratio * (32.0 / max(min_bits, 1))

    assert ratio > 1.05, f"{key} failed compression ratio threshold: got {ratio:.2f}x <= 1.05x"


@pytest.mark.parametrize("key", GROUP4_KEYS)
def test_group4_bert_compression_and_forward(key: str):
    torch.manual_seed(42)
    model = make_bert()
    orig_params = sum(p.numel() for p in model.parameters())

    compressed = apply_compressor(key, copy.deepcopy(model), rank=16)

    # Check forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    out = compressed(input_ids)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        hidden = out[0]
    else:
        hidden = out
    assert torch.isfinite(hidden).all(), f"{key} produced non-finite output on BERT"
    assert hidden.shape == (2, 16, 768)

    # Check compression ratio
    comp_params = sum(p.numel() for p in compressed.parameters())
    zero_params = 0
    total_weights = 0
    for p in compressed.parameters():
        if p.dim() >= 2:
            zero_params += int((p == 0).sum().item())
            total_weights += p.numel()

    sparsity = zero_params / max(total_weights, 1)
    effective_params = comp_params * (1.0 - sparsity)
    ratio = orig_params / max(effective_params, 1)

    has_quant = any(getattr(m, "effective_bits", 32) <= 8 for m in compressed.modules())
    if has_quant:
        min_bits = min(getattr(m, "effective_bits", 32) for m in compressed.modules())
        ratio = ratio * (32.0 / max(min_bits, 1))

    assert ratio > 1.05, f"{key} failed compression ratio threshold: got {ratio:.2f}x <= 1.05x"


GROUP4_RANK_KEYS = [
    "mgie_budget",
    "modegpt_compressor",
    "nmf_compressor",
    "oats_compressor",
    "optimal_subspace_projection",
    "residual_svd_compressor",
    "rpca_compressor",
    "saes_svd_v2",
    "salr",
    "slicegpt",
]


@pytest.mark.parametrize("key", GROUP4_RANK_KEYS)
def test_group4_monotonic_pareto(key: str):
    """Verify that decreasing ranks [256, 128, 64, 32] yield strictly decreasing parameter counts and increasing reduction ratios."""
    ranks = [256, 128, 64, 32]
    prev_params = float("inf")
    prev_ratio = 0.0

    for rank in ranks:
        model = make_mlp()
        orig_params = sum(p.numel() for p in model.parameters())

        comp_model = apply_compressor(key, copy.deepcopy(model), rank=rank)
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
        x = torch.randn(2, 768, requires_grad=True)
        out = comp_model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        assert out.shape == (2, 768), f"Wrong output shape at rank {rank} for {key}"
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

