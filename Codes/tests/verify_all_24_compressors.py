#!/usr/bin/env python3
"""
Comprehensive verification test suite for all 24 Structural Pruning & Sparsity Compression Modules.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from framework.core.registry import COMPRESSOR_REGISTRY


class MiniSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: Any = None, **kwargs: Any) -> Tuple[torch.Tensor, None]:
        B, L, D = x.shape
        Q = self.q_lin(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_lin(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_lin(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, V).transpose(1, 2).reshape(B, L, D)
        out = self.out_lin(context)
        return out, None


class MiniTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, ffn_dim: int = 128) -> None:
        super().__init__()
        self.attention = MiniSelfAttention(hidden_dim=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        attn_out, _ = self.attention(self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn_down(self.act(self.ffn_up(self.norm2(x))))
        return x + ffn_out


class ToyTransformerModel(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layer1 = MiniTransformerBlock(hidden_dim=hidden_dim, ffn_dim=128)
        self.layer2 = MiniTransformerBlock(hidden_dim=hidden_dim, ffn_dim=128)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None, **kwargs: Any) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.classifier(x.mean(dim=1))
        return logits


def run_tests() -> bool:
    print("=" * 80, flush=True)
    print("LEAD STRUCTURAL PRUNING & SPARSITY SPECIALIST - VERIFICATION SUITE", flush=True)
    print("=" * 80, flush=True)

    torch.manual_seed(42)
    dummy_input_ids = torch.randint(0, 255, (2, 16))

    modules_to_test = [
        ("01_slicegpt", "slicegpt", {"sparsity": 0.25, "non_uniform": True}),
        ("02_slicegpt_pca", "slicegpt_pca", {"sparsity": 0.3}),
        ("03_sparsegpt", "sparsegpt", {"sparsity": 0.5, "structured_24": True}),
        ("04_admm_compressor", "admm", {"sparsity": 0.5, "quant_bits": 4, "structured_24": True}),
        ("05_admm_pruner", "admm_pruner", {"sparsity": 0.5, "structured_24": True}),
        ("06_armor_pruner", "armor_pruner", {"block_size": 4, "rank": 8, "structured_24": True}),
        ("07_ast_pruner", "ast_pruner", {"block_size": 4, "n_keep": 2}),
        ("08_bonsai_pruner", "bonsai", {"sparsity": 0.5, "steps": 3, "structured_24": True}),
        ("09_flap", "flap", {"sparsity": 0.2, "metric_type": "WIFV"}),
        ("10_flat_llm", "flat_llm", {"keep_ratio": 0.8}),
        ("11_lillama_compressor", "lillama", {"rank": 16}),
        ("12_mamba_shedder", "mamba_shedder", {"sparsity": 0.25}),
        ("13_proxsparse", "proxsparse", {"sparsity": 0.5, "lambda_prox": 0.01, "steps": 3}),
        ("14_slim_compressor", "slim", {"bits": 4, "sparsity": 0.5, "rank": 8, "structured_24": True}),
        ("15_soka_compressor", "soka", {"rank": 16}),
        ("16_liftr", "liftr", {"replace_attention": True}),
        ("17_lsc_compressor", "lsc", {"rank": 16}),
        ("18_oats_compressor", "oats", {"sparsity": 0.5, "rank_ratio": 0.2, "num_iters": 2}),
        ("19_rotational_slicegpt_int4", "rotational_slicegpt_int4", {"r_ratio": 0.75, "quant_bits": 4}),
        ("20_optimal_subspace_projection", "optimal_subspace_projection", {"rank_ratio": 0.5, "power_iters": 2}),
    ]

    success_count = 0
    fail_count = 0

    for idx, reg_key, kwargs in modules_to_test:
        t0 = time.time()
        try:
            factory = COMPRESSOR_REGISTRY.resolve(reg_key)
            model = ToyTransformerModel()
            compressed_model = factory(model, **kwargs)
            with torch.no_grad():
                out = compressed_model(dummy_input_ids)
            assert out.shape == (2, 2), f"Output shape mismatch: {out.shape} vs expected (2, 2)"
            assert not torch.isnan(out).any(), "Output contains NaN values"
            assert not torch.isinf(out).any(), "Output contains Inf values"

            dt = time.time() - t0
            print(f"  [PASS] {idx.ljust(32)} -> Registry: {reg_key.ljust(28)} ({dt:.3f}s)", flush=True)
            success_count += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"  [FAIL] {idx.ljust(32)} -> Registry: {reg_key.ljust(28)} ERROR: {e} ({dt:.3f}s)", flush=True)
            fail_count += 1

    # Test 21: pruning_utils math routines
    t0 = time.time()
    try:
        from framework.compression_engine.pruning_utils import (
            compute_empirical_hessian,
            apply_24_structured_mask,
            apply_block_mask,
            compute_taylor_first_order_saliency,
            compute_hessian_saliency,
        )
        dummy_act = torch.randn(8, 16, 64)
        H, H_inv = compute_empirical_hessian(dummy_act, damping=0.01)
        assert H.shape == (64, 64) and H_inv.shape == (64, 64)
        assert not torch.isnan(H_inv).any()
        
        W = torch.randn(64, 64)
        W_24, mask_24 = apply_24_structured_mask(W, dim=1)
        assert W_24.shape == (64, 64)
        
        W_blk, mask_blk = apply_block_mask(W, block_size=(4, 4), sparsity=0.5)
        assert W_blk.shape == (64, 64)
        
        s_taylor = compute_taylor_first_order_saliency(W, activations=dummy_act)
        assert s_taylor.shape == (64, 64)
        
        s_hess = compute_hessian_saliency(W, H_inv)
        assert s_hess.shape == (64, 64)

        dt = time.time() - t0
        print(f"  [PASS] {'21_pruning_utils'.ljust(32)} -> Hessian, Saliency, 2:4, Block Masks ({dt:.3f}s)", flush=True)
        success_count += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"  [FAIL] {'21_pruning_utils'.ljust(32)} -> ERROR: {e} ({dt:.3f}s)", flush=True)
        fail_count += 1

    # Test 22: slicegpt_compressor re-export
    t0 = time.time()
    try:
        from framework.compression_engine.slicegpt_compressor import (
            SliceGPTCompressor,
            slicegpt_factory,
            SliceGPTPCACompressor,
            slicegpt_pca_factory,
        )
        assert SliceGPTCompressor is not None
        assert slicegpt_factory is not None
        assert SliceGPTPCACompressor is not None
        assert slicegpt_pca_factory is not None
        dt = time.time() - t0
        print(f"  [PASS] {'22_slicegpt_compressor'.ljust(32)} -> Re-exports Verified ({dt:.3f}s)", flush=True)
        success_count += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"  [FAIL] {'22_slicegpt_compressor'.ljust(32)} -> ERROR: {e} ({dt:.3f}s)", flush=True)
        fail_count += 1

    # Test 23: proxsparse_pruner re-export
    t0 = time.time()
    try:
        from framework.compression_engine.proxsparse_pruner import (
            ProxSparsePruner,
            ProxSparseCompressor,
            proxsparse_factory,
        )
        assert ProxSparsePruner is not None
        assert ProxSparseCompressor is not None
        assert proxsparse_factory is not None
        dt = time.time() - t0
        print(f"  [PASS] {'23_proxsparse_pruner'.ljust(32)} -> Re-exports Verified ({dt:.3f}s)", flush=True)
        success_count += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"  [FAIL] {'23_proxsparse_pruner'.ljust(32)} -> ERROR: {e} ({dt:.3f}s)", flush=True)
        fail_count += 1

    # Test 24: oats re-export
    t0 = time.time()
    try:
        from framework.compression_engine.oats import (
            OATSCompressor,
            oats_compressor_factory,
        )
        assert OATSCompressor is not None
        assert oats_compressor_factory is not None
        dt = time.time() - t0
        print(f"  [PASS] {'24_oats'.ljust(32)} -> Re-exports Verified ({dt:.3f}s)", flush=True)
        success_count += 1
    except Exception as e:
        dt = time.time() - t0
        print(f"  [FAIL] {'24_oats'.ljust(32)} -> ERROR: {e} ({dt:.3f}s)", flush=True)
        fail_count += 1

    total = success_count + fail_count
    print("=" * 80, flush=True)
    print(f"FINAL RESULT: {success_count}/{total} PASSED, {fail_count} FAILED", flush=True)
    print("=" * 80, flush=True)
    return fail_count == 0


if __name__ == "__main__":
    ok = run_tests()
    if not ok:
        sys.exit(1)
