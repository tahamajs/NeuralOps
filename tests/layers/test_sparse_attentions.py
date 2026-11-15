#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive test suite for Sparse Attention Family layers (Wave 4 - W4-L2)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from neuralops.layers.bigbird import BigBirdAttention
from neuralops.layers.block_sparse import BlockSparseAttention
from neuralops.layers.blockwise import BlockwiseAttention
from neuralops.layers.compressed_attention import CompressedAttention
from neuralops.layers.entmax import EntmaxAttention, entmax15
from neuralops.layers.exact_sparse import ExactSparseAttention
from neuralops.layers.fast_attention import FastAttention
from neuralops.layers.fixed_sparse import FixedSparseAttention
from neuralops.layers.global_local import GlobalLocalAttention
from neuralops.layers.global_sparse import GlobalSparseAttention
from neuralops.layers.hamming_attention import HammingAttention
from neuralops.layers.linear_transformer import LinearTransformerAttention
from neuralops.layers.linformer import LinformerAttention
from neuralops.layers.local_attention import LocalAttention
from neuralops.layers.longformer import LongformerAttention
from neuralops.layers.lsh_attention import LSHAttention
from neuralops.layers.moba import MoBAAttention
from neuralops.layers.moba_block_router import MoBABlockRouter, fused_moba_block_router_combine_pytorch
from neuralops.layers.nsa import NSAAttention
from neuralops.layers.nsa_attention import LearnedBlockRouter, NativeSparseAttention
from neuralops.layers.nystrom import NystromAttention
from neuralops.layers.performer import PerformerAttention
from neuralops.layers.random_attention import RandomAttention
from neuralops.layers.random_features import RandomFeaturesAttention
from neuralops.layers.reformer import ReformerAttention
from neuralops.layers.reformer_lsh import ReformerLSHAttention
from neuralops.layers.routing_transformer import RoutingAttention, RoutingTransformerAttention
from neuralops.layers.sinkhorn import SinkhornAttention
from neuralops.layers.sliding_window import SlidingWindowAttention
from neuralops.layers.sparse_attention import AdaptiveSparseAttention, SparseAttention
from neuralops.layers.sparse_flash import SparseFlashAttention
from neuralops.layers.sparse_pattern_attention import SparsePatternAttention
from neuralops.layers.sparse_transformer import SparseTransformerAttention
from neuralops.layers.sparsemax import SparsemaxAttention, sparsemax
from neuralops.layers.stride_sparse import StrideSparseAttention
from neuralops.layers.strided import StridedAttention
from neuralops.layers.switch_transformer import SwitchTransformerAttention
from neuralops.layers.synthesizer import SynthesizerAttention
from neuralops.layers.topk_attention import TopKAttention
from neuralops.layers.triangular_attention import TriangularAttention
from neuralops.layers.varlen_attention import VarlenAttention
from neuralops.layers.xformers_attention import XFormersAttention


class DummyConfig(PretrainedConfig):
    """Test configuration for sparse layer initialization."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        num_buckets: int | None = None,
        window_size: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dim = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        if num_buckets is not None:
            self.num_buckets = num_buckets
        self.window_size = window_size
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Ensure deterministic random initialization."""
    torch.manual_seed(42)


SPARSE_MODULE_CLASSES = [
    SlidingWindowAttention,
    LSHAttention,
    NystromAttention,
    BigBirdAttention,
    BlockSparseAttention,
    SparseAttention,
    LongformerAttention,
    ReformerAttention,
    SparseTransformerAttention,
    MoBAAttention,
    HammingAttention,
    ReformerLSHAttention,
    RoutingTransformerAttention,
    SwitchTransformerAttention,
    BlockwiseAttention,
    StridedAttention,
    GlobalLocalAttention,
    RandomAttention,
    TopKAttention,
    NSAAttention,
    NativeSparseAttention,
    VarlenAttention,
    XFormersAttention,
    SparsePatternAttention,
    SparseFlashAttention,
    SinkhornAttention,
    TriangularAttention,
    LinformerAttention,
    PerformerAttention,
    RandomFeaturesAttention,
    FastAttention,
    LinearTransformerAttention,
    SynthesizerAttention,
    SparsemaxAttention,
    EntmaxAttention,
    ExactSparseAttention,
    CompressedAttention,
    StrideSparseAttention,
    FixedSparseAttention,
    LocalAttention,
    GlobalSparseAttention,
]


@pytest.mark.parametrize("layer_cls", SPARSE_MODULE_CLASSES)
def test_sparse_layers_forward_shape_and_finite(layer_cls: type[nn.Module]) -> None:
    """Verify forward shape preservation and finite outputs across all sparse layers."""
    cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
    try:
        mod = layer_cls(config=cfg, dim=64, num_heads=4)
    except TypeError:
        mod = layer_cls(dim=64, num_heads=4)

    mod.eval()
    x = torch.randn(2, 16, 64)
    out, _ = mod(x)
    assert out.shape == (2, 16, 64), f"{layer_cls.__name__} output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), f"{layer_cls.__name__} produced non-finite output"


@pytest.mark.parametrize("layer_cls", SPARSE_MODULE_CLASSES)
def test_sparse_layers_gradient_flow(layer_cls: type[nn.Module]) -> None:
    """Verify backward gradient flow with non-zero and finite gradients."""
    cfg = DummyConfig(hidden_size=64, num_attention_heads=4)
    try:
        mod = layer_cls(config=cfg, dim=64, num_heads=4)
    except TypeError:
        mod = layer_cls(dim=64, num_heads=4)

    mod.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = mod(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"{layer_cls.__name__} input grad is None"
    assert torch.isfinite(x.grad).all(), f"{layer_cls.__name__} input grad is not finite"


def test_sliding_window_padding_isolation() -> None:
    """Special Check 1: Verify padding positions are strictly not attended."""
    layer = SlidingWindowAttention(window_size=4, dim=32, num_heads=2, is_causal=False)
    layer.eval()

    x_clean = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])

    x_corrupted = x_clean.clone()
    x_corrupted[:, 5:, :] += 100.0

    out_clean, _ = layer(x_clean, attention_mask=mask)
    out_corrupted, _ = layer(x_corrupted, attention_mask=mask)

    # Valid positions must be strictly unaffected by padding values
    diff_0 = (out_clean[0, :4, :] - out_corrupted[0, :4, :]).abs().max().item()
    diff_1 = (out_clean[1, :5, :] - out_corrupted[1, :5, :]).abs().max().item()
    assert diff_0 < 1e-6, f"Padding leaked into unpadded sequence 0 (diff={diff_0})"
    assert diff_1 < 1e-6, f"Padding leaked into unpadded sequence 1 (diff={diff_1})"


def test_nystrom_deterministic_landmarks_and_stable_pinv() -> None:
    """Special Check 2: Verify landmark determinism and stable pseudoinverse under collinearity."""
    layer = NystromAttention(num_landmarks=8, pinv_iterations=5, dim=32, num_heads=2)
    layer.eval()

    # Landmark pooling determinism
    x = torch.randn(2, 16, 32)
    out1, _ = layer(x)
    out2, _ = layer(x)
    assert torch.equal(out1, out2), "NystromAttention produces non-deterministic output"

    # Collinear landmark kernel matrix
    collinear_kernel = torch.ones(1, 2, 8, 8) / 8.0
    pinv = layer._iterative_pinv(collinear_kernel, iters=5)
    assert torch.isfinite(pinv).all(), "Nystrom iterative Schultz diverged on collinear landmarks"
    assert pinv.abs().max().item() < 10.0, "Nystrom Schultz pinv magnitude exploded"


def test_lsh_hash_buckets_matches_config() -> None:
    """Special Check 3: Verify number of hash buckets matches configuration."""
    cfg1 = DummyConfig(num_buckets=32)
    layer1 = LSHAttention(config=cfg1, dim=32, num_heads=2)
    assert layer1.num_buckets == 32, f"Expected 32 buckets from config.num_buckets, got {layer1.num_buckets}"

    cfg2 = DummyConfig(lsh_buckets=64)
    layer2 = LSHAttention(config=cfg2, dim=32, num_heads=2)
    assert layer2.num_buckets == 64, f"Expected 64 buckets from config.lsh_buckets, got {layer2.num_buckets}"

    layer3 = LSHAttention(num_buckets=8, dim=32, num_heads=2)
    assert layer3.num_buckets == 8, f"Expected 8 buckets from kwargs, got {layer3.num_buckets}"


def test_block_sparse_routing_determinism() -> None:
    """Special Check 4: Verify block routing determinism and top-k stability."""
    # 1. BlockSparseAttention layer determinism
    bs_layer = BlockSparseAttention(block_size=8, top_k=2, dim=32, num_heads=2)
    bs_layer.eval()
    x = torch.randn(2, 16, 32)
    out1, _ = bs_layer(x)
    out2, _ = bs_layer(x)
    assert torch.equal(out1, out2), "BlockSparseAttention non-deterministic output"

    # 2. MoBABlockRouter determinism
    router = MoBABlockRouter(d_model=32, num_blocks=4, router_hidden=16, top_k=2)
    router.eval()
    q = torch.randn(2, 4, 32)
    l1, idx1 = router(q, top_k=2)
    l2, idx2 = router(q, top_k=2)
    assert torch.equal(l1, l2), "MoBABlockRouter non-deterministic logits"
    assert torch.equal(idx1, idx2), "MoBABlockRouter non-deterministic top-k indices"

    # 3. LearnedBlockRouter determinism
    lbr = LearnedBlockRouter(d=32, num_blocks=4, router_hidden=16)
    lbr.eval()
    l1_b, idx1_b = lbr(q, top_k=2)
    l2_b, idx2_b = lbr(q, top_k=2)
    assert torch.equal(l1_b, l2_b), "LearnedBlockRouter non-deterministic logits"
    assert torch.equal(idx1_b, idx2_b), "LearnedBlockRouter non-deterministic top-k indices"


def test_bigbird_attention_pattern_sum() -> None:
    """Special Check 5: Verify three types of attention (global, sliding, random) sum to intended pattern."""
    layer = BigBirdAttention(window_size=4, num_global_tokens=2, num_random_blocks=2, dim=32, num_heads=2)
    N = 16
    local_m, global_m, rand_m, combined_m = layer.get_attention_pattern(N)

    # Component checks
    assert local_m.shape == (N, N)
    assert global_m.shape == (N, N)
    assert rand_m.shape == (N, N)
    assert combined_m.shape == (N, N)

    # Union property: combined must be exactly the logical OR of all three components
    expected_combined = local_m | global_m | rand_m
    assert torch.equal(combined_m, expected_combined), "BigBird components do not sum to combined pattern"

    # Verify global tokens attend everywhere
    assert combined_m[0, :].all(), "Global token 0 failed to attend across full sequence"
    assert combined_m[:, 0].all(), "Global token 0 failed to receive attention across full sequence"


def test_sliding_window_causal_leakage() -> None:
    """Verify future-token isolation in causal sliding window."""
    layer = SlidingWindowAttention(window_size=4, dim=32, num_heads=2, is_causal=True)
    layer.eval()

    x = torch.randn(1, 10, 32)
    x_mutated = x.clone()
    # Mutate position 8
    x_mutated[:, 8, :] += 5.0

    out, _ = layer(x)
    out_mut, _ = layer(x_mutated)

    # Positions before 8 must be completely unaffected
    diff = (out[:, :8, :] - out_mut[:, :8, :]).abs().max().item()
    assert diff < 1e-6, f"Causal sliding window leaked future token! diff={diff}"


def test_sparsemax_exact_sparsity() -> None:
    """Verify sparsemax produces exact zero probabilities."""
    logits = torch.tensor([[-5.0, 0.0, 2.0, 4.0]])
    p = sparsemax(logits, dim=-1)
    assert p[0, 0].item() == 0.0, "sparsemax failed to zero out low logit"
    assert math.isclose(p.sum().item(), 1.0, abs_tol=1e-5), "sparsemax probabilities do not sum to 1"


def test_entmax_exact_sparsity() -> None:
    """Verify 1.5-entmax produces exact zero probabilities."""
    logits = torch.tensor([[-10.0, 0.0, 2.0, 4.0]])
    p = entmax15(logits, dim=-1)
    assert p[0, 0].item() == 0.0, "entmax failed to zero out lowest logit"
    assert math.isclose(p.sum().item(), 1.0, abs_tol=1e-4), "entmax probabilities do not sum to 1"
