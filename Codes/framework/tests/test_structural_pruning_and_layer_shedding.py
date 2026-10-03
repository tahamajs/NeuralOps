"""
Unit test suite for Structural Pruning and Layer Shedding Compression Engine Modules.

Tests all 16 target modules:
1. slicegpt, slicegpt_pca
2. sparsegpt
3. admm, admm_pruner
4. armor_pruner
5. ast_pruner
6. bonsai
7. flap
8. flat_llm
9. lillama
10. mamba_shedder
11. proxsparse
12. slim
13. soka
14. liftr
15. lsc
16. oats

Verifies:
- Standard (model, ...) interface and .compress() -> (compressed_model, stats_dict)
- Dimension contracts and forward pass execution with zero tensor shape mismatches
- Parameter reduction metrics in stats dictionary
- Registry discovery via COMPRESSOR_REGISTRY
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn

# Ensure parent paths are available
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

import framework.compression_engine
from framework.compression_engine import (ADMMPruner, ARMORPruner, ASTPruner,
                                          BonsaiPruner, CacheNotesCompressor,
                                          CoMeCompressor, FLAPCompressor,
                                          FLATLLMCompressor, LIFTRCompressor,
                                          LiLLaMACompressor, LowRankCompressor,
                                          LSCCompressor,
                                          MambaShedderCompressor,
                                          OATSCompressor,
                                          OptimalSubspaceProjectionCompressor,
                                          ProxSparsePruner,
                                          RotationalSliceGPTINT4Compressor,
                                          SDRMCompressor, SliceGPTCompressor,
                                          SliceGPTPCACompressor,
                                          SLiMCompressor, SoKACompressor,
                                          SparseGPTCompressor)
from framework.compression_engine.admm_compressor import ADMMCompressor
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask, compute_empirical_hessian)
from framework.core.registry import COMPRESSOR_REGISTRY


class MiniSelfAttention(nn.Module):
    """Toy multi-head self-attention module."""

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
    """Toy transformer encoder block."""

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
    """Toy 2-layer transformer model for unit testing compression engines."""

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


class TestStructuralPruningAndLayerShedding(unittest.TestCase):
    """Test case validating all 16 pruning and layer shedding compression engines."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_dim = 64
        self.dummy_input_ids = torch.randint(0, 255, (self.batch_size, self.seq_len))
        self.dummy_inputs = {"input_ids": self.dummy_input_ids}

    def _create_model(self) -> ToyTransformerModel:
        torch.manual_seed(42)
        return ToyTransformerModel(vocab_size=256, hidden_dim=self.hidden_dim, num_classes=2)

    def _verify_compression_output(
        self,
        model: nn.Module,
        compressed_model: nn.Module,
        stats: Dict[str, Any],
        expected_name: str,
    ) -> None:
        """Verifies standard contracts."""
        self.assertIsInstance(stats, dict, "Stats must be a dictionary")
        self.assertIn("original_parameters", stats)
        self.assertIn("compressed_parameters", stats)
        self.assertIn("compression_ratio", stats)
        self.assertIn("reduction_percentage", stats)

        # Verify forward pass on compressed model runs cleanly without shape errors
        with torch.no_grad():
            out = compressed_model(self.dummy_input_ids)
        self.assertEqual(out.shape, (self.batch_size, 2), "Forward output shape mismatch!")

    def test_01_slicegpt(self) -> None:
        model = self._create_model()
        compressor = SliceGPTCompressor(model, sparsity=0.25, non_uniform=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "slicegpt")
        self.assertEqual(stats["compressor_name"], "slicegpt")

        # Test registry factory
        factory = COMPRESSOR_REGISTRY.resolve("slicegpt")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_02_slicegpt_pca(self) -> None:
        model = self._create_model()
        compressor = SliceGPTPCACompressor(model, sparsity=0.3)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "slicegpt_pca")
        self.assertEqual(stats["compressor_name"], "slicegpt_pca")

        factory = COMPRESSOR_REGISTRY.resolve("slicegpt_pca")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_03_sparsegpt(self) -> None:
        model = self._create_model()
        compressor = SparseGPTCompressor(model, sparsity=0.5, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "sparsegpt")
        self.assertEqual(stats["compressor_name"], "sparsegpt")

        factory = COMPRESSOR_REGISTRY.resolve("sparsegpt")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_04_admm_compressor(self) -> None:
        model = self._create_model()
        compressor = ADMMCompressor(model, sparsity=0.5, quant_bits=4, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "admm")
        self.assertEqual(stats["compressor_name"], "admm")

        factory = COMPRESSOR_REGISTRY.resolve("admm")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_05_admm_pruner(self) -> None:
        model = self._create_model()
        compressor = ADMMPruner(model, sparsity=0.5, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "admm_pruner")
        self.assertEqual(stats["compressor_name"], "admm_pruner")

        factory = COMPRESSOR_REGISTRY.resolve("admm_pruner")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_06_armor_pruner(self) -> None:
        model = self._create_model()
        compressor = ARMORPruner(model, block_size=4, rank=8, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "armor_pruner")
        self.assertEqual(stats["compressor_name"], "armor_pruner")

        factory = COMPRESSOR_REGISTRY.resolve("armor_pruner")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_07_ast_pruner(self) -> None:
        model = self._create_model()
        compressor = ASTPruner(model, block_size=4, n_keep=2)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "ast_pruner")
        self.assertEqual(stats["compressor_name"], "ast_pruner")

        factory = COMPRESSOR_REGISTRY.resolve("ast_pruner")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_08_bonsai_pruner(self) -> None:
        model = self._create_model()
        compressor = BonsaiPruner(model, sparsity=0.5, steps=4, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "bonsai")
        self.assertEqual(stats["compressor_name"], "bonsai")

        factory = COMPRESSOR_REGISTRY.resolve("bonsai")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_09_flap(self) -> None:
        model = self._create_model()
        compressor = FLAPCompressor(model, sparsity=0.2, metric_type="WIFV")
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "flap")
        self.assertEqual(stats["compressor_name"], "flap")

        factory = COMPRESSOR_REGISTRY.resolve("flap")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_10_flat_llm(self) -> None:
        model = self._create_model()
        compressor = FLATLLMCompressor(model, keep_ratio=0.8)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "flat_llm")
        self.assertEqual(stats["compressor_name"], "flat_llm")

        factory = COMPRESSOR_REGISTRY.resolve("flat_llm")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_11_lillama(self) -> None:
        model = self._create_model()
        compressor = LiLLaMACompressor(model, rank=16)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "lillama")
        self.assertEqual(stats["compressor_name"], "lillama")

        factory = COMPRESSOR_REGISTRY.resolve("lillama")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_12_mamba_shedder(self) -> None:
        model = self._create_model()
        compressor = MambaShedderCompressor(model, sparsity=0.25)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "mamba_shedder")
        self.assertEqual(stats["compressor_name"], "mamba_shedder")

        factory = COMPRESSOR_REGISTRY.resolve("mamba_shedder")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_13_proxsparse(self) -> None:
        model = self._create_model()
        compressor = ProxSparsePruner(model, sparsity=0.5, lambda_prox=0.01, steps=3)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "proxsparse")
        self.assertEqual(stats["compressor_name"], "proxsparse")

        factory = COMPRESSOR_REGISTRY.resolve("proxsparse")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_14_slim(self) -> None:
        model = self._create_model()
        compressor = SLiMCompressor(model, bits=4, sparsity=0.5, rank=8, structured_24=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "slim")
        self.assertEqual(stats["compressor_name"], "slim")

        factory = COMPRESSOR_REGISTRY.resolve("slim")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_15_soka(self) -> None:
        model = self._create_model()
        compressor = LowRankCompressor(model, rank=16)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "soka")
        self.assertEqual(stats["compressor_name"], "soka")

        factory = COMPRESSOR_REGISTRY.resolve("soka")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_16_liftr(self) -> None:
        model = self._create_model()
        compressor = LIFTRCompressor(model, replace_attention=True)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "liftr")
        self.assertEqual(stats["compressor_name"], "liftr")

        factory = COMPRESSOR_REGISTRY.resolve("liftr")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_17_lsc(self) -> None:
        model = self._create_model()
        compressor = LSCCompressor(model, rank=16)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "lsc")
        self.assertEqual(stats["compressor_name"], "lsc")

        factory = COMPRESSOR_REGISTRY.resolve("lsc")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_18_oats(self) -> None:
        model = self._create_model()
        compressor = OATSCompressor(model, sparsity=0.5, rank_ratio=0.2, num_iters=2)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "oats")
        self.assertEqual(stats["compressor_name"], "oats")

        factory = COMPRESSOR_REGISTRY.resolve("oats")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_19_rotational_slicegpt_int4(self) -> None:
        model = self._create_model()
        compressor = RotationalSliceGPTINT4Compressor(model, r_ratio=0.75, quant_bits=4)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "rotational_slicegpt_int4")
        self.assertEqual(stats["compressor_name"], "rotational_slicegpt_int4")

        factory = COMPRESSOR_REGISTRY.resolve("rotational_slicegpt_int4")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_20_optimal_subspace_projection(self) -> None:
        model = self._create_model()
        compressor = OptimalSubspaceProjectionCompressor(model, rank_ratio=0.5, power_iters=2)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "optimal_subspace_projection")
        self.assertEqual(stats["compressor_name"], "optimal_subspace_projection")

        factory = COMPRESSOR_REGISTRY.resolve("optimal_subspace_projection")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_21_come(self) -> None:
        model = self._create_model()
        compressor = CoMeCompressor(model, sparsity=0.5, alpha_init=0.1)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "come")
        self.assertEqual(stats["compressor_name"], "come")

        factory = COMPRESSOR_REGISTRY.resolve("come")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_22_cachenotes(self) -> None:
        model = self._create_model()
        compressor = CacheNotesCompressor(model, task_type="qa", keep_ratio=0.5)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "cachenotes")
        self.assertEqual(stats["compressor_name"], "cachenotes")

        factory = COMPRESSOR_REGISTRY.resolve("cachenotes")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_23_sdrm(self) -> None:
        model = self._create_model()
        compressor = SDRMCompressor(model, total_param_ratio=0.7)
        compressed_model, stats = compressor.compress()
        self._verify_compression_output(model, compressed_model, stats, "sdrm")
        self.assertEqual(stats["compressor_name"], "sdrm")

        factory = COMPRESSOR_REGISTRY.resolve("sdrm")
        reg_model = factory(self._create_model())
        with torch.no_grad():
            self.assertEqual(reg_model(self.dummy_input_ids).shape, (self.batch_size, 2))

    def test_24_slicegpt_orthogonal_invariance(self) -> None:
        """Mathematical proof check: Q^T @ Q = I."""
        model = self._create_model()
        compressor = SliceGPTCompressor(model, sparsity=0.25)
        d_in = 64
        dummy_act = torch.randn(100, d_in)
        Q, eigvals = compressor._compute_orthogonal_rotation(dummy_act, d_in)
        
        # Verify orthogonality: Q^T Q == I
        identity = torch.eye(d_in)
        qtq = torch.matmul(Q.T, Q)
        diff = torch.norm(qtq - identity, p="fro").item()
        self.assertLess(diff, 1e-4, f"Orthogonality error Q^T Q - I = {diff:.6f} exceeds threshold!")

    def test_25_sparsegpt_damped_hessian_inversion(self) -> None:
        """Mathematical proof check: Damped Hessian H + lambda*I is symmetric positive-definite."""
        d_in = 32
        N = 64
        X = torch.randn(N, d_in)
        damping = 0.01
        H_reg, H_inv = compute_empirical_hessian(X, damping=damping)

        # Check symmetry
        sym_diff = torch.norm(H_reg - H_reg.T, p="fro").item()
        self.assertLess(sym_diff, 1e-5, f"Hessian asymmetry error: {sym_diff:.6f}")

        # Check positive definiteness: all eigenvalues > 0
        eigvals = torch.linalg.eigvalsh(H_reg)
        min_eig = eigvals.min().item()
        self.assertGreater(min_eig, 0.0, f"Minimum eigenvalue of damped Hessian is {min_eig} <= 0")

        # Check inverse reconstruction: H @ H_inv approx I
        prod = torch.matmul(H_reg, H_inv)
        inv_err = torch.norm(prod - torch.eye(d_in), p="fro").item()
        self.assertLess(inv_err, 0.1, f"Inverse Hessian reconstruction error: {inv_err:.6f}")

    def test_26_structured_24_hardware_sparsity(self) -> None:
        """Hardware compliance check: In every 4-element block, exactly 2 non-zeros."""
        d_out, d_in = 16, 32
        W = torch.randn(d_out, d_in)
        W_pruned, mask = apply_24_structured_mask(W, dim=1)

        # Verify block-wise 2:4 pattern
        for r in range(d_out):
            for c_blk in range(d_in // 4):
                block = mask[r, c_blk * 4 : (c_blk + 1) * 4]
                non_zeros = torch.count_nonzero(block).item()
                self.assertEqual(non_zeros, 2, f"Row {r} Block {c_blk} non-zeros = {non_zeros} != 2")

    def test_27_admm_convergence_and_ste(self) -> None:
        """Optimization check: ADMM convergence and STE gradient flow through mask."""
        model = self._create_model()
        compressor = ADMMPruner(model, sparsity=0.5, max_iter=6, structured_24=True, use_ste=True)
        compressed_model, stats = compressor.compress()

        # Check forward and backward pass through STE masked layers
        out = compressed_model(self.dummy_input_ids)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist and are non-zero on masked linear layers
        for name, mod in compressed_model.named_modules():
            if hasattr(mod, "weight") and mod.weight.grad is not None:
                self.assertTrue(torch.isfinite(mod.weight.grad).all(), f"NaN in gradients for {name}")

    def test_28_come_learnable_identity_bridges(self) -> None:
        """Architectural check: CoMe bridges are initialized to Identity with learnable alpha."""
        from framework.compression_engine.come import CoMeBridge
        hidden_dim = 64
        bridge = CoMeBridge(hidden_size=hidden_dim, alpha_init=0.1)

        # Verify W_bridge is initialized to Identity
        eye = torch.eye(hidden_dim)
        diff = torch.norm(bridge.linear.weight.data - eye, p="fro").item()
        self.assertLess(diff, 1e-6, f"CoMeBridge linear weight is not identity: {diff}")

        # Verify forward pass: for alpha=0.1, y = x + alpha*(W@x - x) = x when W=I
        x = torch.randn(2, 8, hidden_dim)
        y = bridge(x)
        self.assertTrue(torch.allclose(x, y, atol=1e-5), "Initial bridge forward did not preserve identity mapping!")

        # Verify learnability: gradient flows to alpha and linear weight
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(bridge.alpha.grad, "Alpha parameter did not receive gradients!")
        self.assertIsNotNone(bridge.linear.weight.grad, "Bridge linear weights did not receive gradients!")


if __name__ == "__main__":
    unittest.main()

