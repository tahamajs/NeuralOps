"""
Comprehensive Audit & Evaluation Suite for SVD and Low-Rank Attention Mechanisms & Compression Engines.

Audited Modules:
1. attentions/svd_linear_attention.py (SVDLinear, SVDLinearAttention, exact, linear, decay gating, step_recurrent)
2. attentions/adaptive_svd_attention.py (AdaptiveSVDLinear, AdaptiveSVDAttention, energy thresholding)
3. attentions/asvd_attention.py (ASVDLinear, ASVDAttention)
4. attentions/linformer_attention.py (LinformerAttention, sequence projection)
5. compression_engine/res_svd.py (ResSVDLinear, ResSVDCompressor, SpectralResidualLoRA)
6. compression_engine/adaptive_ranks.py (AdaptiveRanksCompressor, covariance whitening, energy thresholding)
7. compression_engine/direct_svd.py (DirectSVDCompressor, symmetrical SVD)
8. compression_engine/dobi_svd.py (DobiSVDLinearLayer, DobiSVDCompressor, spectral gating)
9. compression_engine/erc_svd.py (ERCCompressor, Frobenius error bound)
10. compression_engine/entropy_budget_allocation.py (EntropyBudgetCompressor, singular entropy allocation)
11. compression_engine/kronecker_tt_lora.py (KroneckerTTLoRALayer, KroneckerTTLoRACompressor, TT cores, LoRA bypass)
"""

import math
import os
import sys
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure Codes/ is in sys.path
sys_path_codes = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys_path_codes not in sys.path:
    sys.path.insert(0, sys_path_codes)

from framework.attentions.adaptive_svd_attention import (AdaptiveSVDAttention,
                                                         AdaptiveSVDLinear)
from framework.attentions.asvd_attention import ASVDAttention, ASVDLinear
from framework.attentions.linformer_attention import LinformerAttention
from framework.attentions.svd_linear_attention import (
    SVDLinear, SVDLinearAttention, compute_exponential_decay)
from framework.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor, adaptive_ranks_factory)
from framework.compression_engine.direct_svd import (DirectSVDCompressor,
                                                     direct_svd_factory)
from framework.compression_engine.dobi_svd import (DobiSVDCompressor,
                                                   DobiSVDLinearLayer,
                                                   dobi_factory)
from framework.compression_engine.entropy_budget_allocation import (
    EntropyBudgetCompressor, entropy_budget_factory)
from framework.compression_engine.erc_svd import ERCCompressor, erc_factory
from framework.compression_engine.kronecker_tt_lora import (
    KroneckerTTLoRACompressor, KroneckerTTLoRALayer, kronecker_tt_lora_factory)
from framework.compression_engine.res_svd import (ResSVDCompressor,
                                                  ResSVDLinear,
                                                  SpectralResidualLoRA,
                                                  res_svd_factory)


class MockMHA(nn.Module):
    """Mock standard Multi-Head Attention layer for weight copying tests."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)


class MockTransformerModel(nn.Module):
    """Mock multi-layer transformer model for compression engine testing."""

    def __init__(self, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_size, hidden_size),
                "k_proj": nn.Linear(hidden_size, hidden_size),
                "v_proj": nn.Linear(hidden_size, hidden_size),
                "out_proj": nn.Linear(hidden_size, hidden_size),
                "fc1": nn.Linear(hidden_size, hidden_size * 2),
                "fc2": nn.Linear(hidden_size * 2, hidden_size),
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, 2)  # Should be excluded from compression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            q = layer["q_proj"](x)
            k = layer["k_proj"](x)
            v = layer["v_proj"](x)
            attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(x.size(-1))
            attn_out = torch.matmul(F.softmax(attn, dim=-1), v)
            x = x + layer["out_proj"](attn_out)
            ffn = layer["fc2"](F.gelu(layer["fc1"](x)))
            x = x + ffn
        return self.classifier(x[:, 0, :])


def get_dummy_config(hidden_size: int = 768, num_heads: int = 12, rank: int = 64):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_heads=num_heads,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        intermediate_size=hidden_size * 4,
        rank=rank,
        svd_rank=rank,
        asvd_rank=rank,
        residual_rank=8,
        energy_threshold=0.92,
        linear_attention_eps=1e-6,
    )


class TestLeadSVDSpecialistAudit(unittest.TestCase):
    """Lead SVD & Factorization Specialist Comprehensive Audit Test Suite."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.num_heads = 12
        self.config = get_dummy_config(self.hidden_size, self.num_heads, rank=64)

    # =========================================================================
    # 1. SVDLinear & SVDLinearAttention
    # =========================================================================
    def test_svd_linear_forward_backward_and_stiefel(self):
        """Audit SVDLinear layer forward/backward, LoRA residual, and Stiefel orthogonality loss."""
        layer = SVDLinear(
            in_features=768,
            out_features=768,
            rank=64,
            bias=True,
            freeze_svd=False,
            lora_rank=8,
            lora_alpha=16.0,
        )
        x = torch.randn(2, 32, 768, requires_grad=True)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, 768))
        self.assertFalse(torch.isnan(out).any())

        # Test Stiefel manifold loss
        ortho_loss = layer.get_orthogonality_loss()
        self.assertTrue(torch.is_tensor(ortho_loss))
        self.assertGreaterEqual(ortho_loss.item(), 0.0)

        # Backward pass
        total_loss = out.sum() + ortho_loss
        total_loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertIsNotNone(layer.U.grad)
        self.assertIsNotNone(layer.V.grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)

    def test_svd_linear_frozen_base(self):
        """Audit SVDLinear with frozen SVD base and active LoRA adapter."""
        layer = SVDLinear(
            in_features=768,
            out_features=768,
            rank=64,
            bias=True,
            freeze_svd=True,
            lora_rank=8,
        )
        x = torch.randn(2, 16, 768, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNone(layer.U.grad)
        self.assertIsNone(layer.V.grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)

    def test_svd_linear_attention_modes(self):
        """Audit SVDLinearAttention across exact, linear kernel, and decay-gated linear modes."""
        for mode, use_decay in [("exact", False), ("linear", False), ("linear", True)]:
            attn = SVDLinearAttention(
                self.config,
                rank=64,
                mode=mode,
                use_decay_gating=use_decay,
            )
            x = torch.randn(2, 16, self.hidden_size, requires_grad=True)
            mask = torch.ones(2, 16)
            mask[:, 12:] = 0

            out, probs = attn(hidden_states=x, attention_mask=mask, output_attentions=(mode == "exact"))
            self.assertEqual(out.shape, (2, 16, self.hidden_size))
            self.assertFalse(torch.isnan(out).any())
            if mode == "exact":
                self.assertIsNotNone(probs)
                self.assertEqual(probs.shape, (2, self.num_heads, 16, 16))

            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertFalse(torch.isnan(x.grad).any())

    def test_svd_linear_decay_step_recurrent(self):
        """Audit step_recurrent streaming generation vs sequential forward in DecayGatedLinearAttention."""
        attn = SVDLinearAttention(
            self.config,
            rank=64,
            mode="decay",
            use_decay_gating=True,
        )
        attn.eval()
        x = torch.randn(1, 4, self.hidden_size)

        # Full sequential forward
        out_seq, _, (S_final, Z_final) = attn(hidden_states=x, return_state=True)

        # Step-by-step streaming recurrent forward
        state = None
        out_steps = []
        for t in range(4):
            x_t = x[:, t : t + 1, :]
            out_t, state = attn.step_recurrent(x_t, past_state=state)
            out_steps.append(out_t)

        out_streamed = torch.cat(out_steps, dim=1)
        # Verify numerical equivalence between full sequence recurrence and step-by-step streaming
        max_diff = torch.max(torch.abs(out_seq - out_streamed)).item()
        self.assertLess(max_diff, 1e-5, f"Streaming recurrent output diverges from sequential: max_diff={max_diff}")

    # =========================================================================
    # 2. AdaptiveSVDAttention & AdaptiveSVDLinear
    # =========================================================================
    def test_adaptive_svd_attention_forward_backward_energy(self):
        """Audit AdaptiveSVDAttention forward/backward, cumulative energy thresholding, and residual adapter."""
        attn = AdaptiveSVDAttention(
            self.config,
            svd_rank=64,
            residual_rank=8,
            energy_threshold=0.92,
        )
        # Test copy_weights from pretrained MHA
        mock_mha = MockMHA(self.hidden_size, self.num_heads)
        attn.copy_weights(mock_mha)

        x = torch.randn(2, 24, self.hidden_size, requires_grad=True)
        mask = torch.ones(2, 24)
        mask[:, 20:] = 0

        out, probs = attn(hidden_states=x, attention_mask=mask, output_attentions=True)
        self.assertEqual(out.shape, (2, 24, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())
        self.assertIsNotNone(probs)

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Test energy rank selection function
        S = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05])
        r = attn._compute_energy_rank(S, max_rank=8, min_rank=1)
        self.assertEqual(r, 2)

    # =========================================================================
    # 3. ASVDAttention & ASVDLinear
    # =========================================================================
    def test_asvd_attention_forward_backward(self):
        """Audit ASVDAttention with explicit singular value vector S."""
        attn = ASVDAttention(self.config, rank=64, lora_rank=0)
        mock_mha = MockMHA(self.hidden_size, self.num_heads)
        attn.copy_weights(mock_mha)

        x = torch.randn(2, 16, self.hidden_size, requires_grad=True)
        out, probs = attn(hidden_states=x, output_attentions=False)
        self.assertEqual(out.shape, (2, 16, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertIsNotNone(attn.query.S.grad)

    # =========================================================================
    # 4. LinformerAttention
    # =========================================================================
    def test_linformer_attention_forward_backward(self):
        """Audit LinformerAttention with sequence dimension projection matrices E and F."""
        attn = LinformerAttention(self.config, k=32, max_seq_len=128)
        mock_mha = MockMHA(self.hidden_size, self.num_heads)
        attn.copy_weights(mock_mha)

        x = torch.randn(2, 48, self.hidden_size, requires_grad=True)
        mask = torch.ones(2, 48)
        mask[:, 40:] = 0

        out, probs = attn(hidden_states=x, attention_mask=mask)
        self.assertEqual(out.shape, (2, 48, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertIsNotNone(attn.E.grad)
        self.assertIsNotNone(attn.F.grad)

    # =========================================================================
    # 5. Res-SVD / Spectral Residual LoRA
    # =========================================================================
    def test_res_svd_linear_and_compressor(self):
        """Audit ResSVDLinear residual tail initialization, forward/backward, and merge_weights."""
        dense = nn.Linear(128, 128)
        res_layer = ResSVDLinear.from_linear(
            linear=dense,
            base_rank=32,
            residual_rank=8,
            scaling=1.0,
            factorize_base=True,
            freeze_base=True,
        )
        self.assertEqual(res_layer.base_rank, 32)
        self.assertEqual(res_layer.residual_rank, 8)
        self.assertFalse(res_layer.base_U.requires_grad)
        self.assertTrue(res_layer.lora_A.requires_grad)

        # Forward / Backward
        x = torch.randn(2, 10, 128, requires_grad=True)
        out = res_layer(x)
        self.assertEqual(out.shape, (2, 10, 128))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(res_layer.base_U.grad)
        self.assertIsNotNone(res_layer.lora_A.grad)

        # Reconstruction error norm at init
        err = res_layer.get_residual_error_norm(dense.weight.data)
        self.assertLess(err, 1.0)

        # Merged weights
        merged = res_layer.merge_weights()
        self.assertIsInstance(merged, nn.Linear)
        diff = torch.norm(res_layer(x.detach()) - merged(x.detach())).item()
        self.assertLess(diff, 1e-4)

        # Compressor test
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        comp_model, stats = ResSVDCompressor(model, base_rank=32, residual_rank=8).compress()
        self.assertIn("trainable_parameters", stats)
        self.assertIn("frozen_parameters", stats)
        out_comp = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out_comp.shape, (2, 2))

    # =========================================================================
    # 6. AdaptiveRanksCompressor
    # =========================================================================
    def test_adaptive_ranks_compressor_and_energy(self):
        """Audit AdaptiveRanksCompressor cumulative energy thresholding (tau=0.92) and rank floor."""
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.92,
            min_rank=16,
        )
        comp_model, stats = compressor.compress()
        out = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(torch.isnan(out).any())

        for name, module in comp_model.named_modules():
            if hasattr(module, "rank"):
                self.assertGreaterEqual(module.rank, 16)

    # =========================================================================
    # 7. DirectSVDCompressor
    # =========================================================================
    def test_direct_svd_compressor(self):
        """Audit DirectSVDCompressor symmetric factor allocation U sqrt(S), sqrt(S) V^T."""
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        compressor = DirectSVDCompressor(model, rank=32, min_rank=16)
        comp_model, stats = compressor.compress()
        out = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(torch.isnan(out).any())
        self.assertLess(stats["compressed_parameters"], stats["original_parameters"])

    # =========================================================================
    # 8. DobiSVDCompressor & DobiSVDLinearLayer
    # =========================================================================
    def test_dobi_svd_linear_and_compressor(self):
        """Audit Dobi-SVD differentiable spectral gating and forward/backward gradient flow."""
        dense = nn.Linear(128, 128)
        dobi_layer, err, r = DobiSVDCompressor(nn.Sequential(dense), rank=32).compress_linear(dense, rank=32)
        self.assertEqual(dobi_layer.rank, 32)
        self.assertIsNotNone(dobi_layer.gate)

        x = torch.randn(2, 10, 128, requires_grad=True)
        out = dobi_layer(x)
        self.assertEqual(out.shape, (2, 10, 128))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(dobi_layer.gate.grad)
        self.assertIsNotNone(dobi_layer.U.grad)
        self.assertIsNotNone(dobi_layer.V.grad)

        # Full model compressor test
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        comp_model, stats = DobiSVDCompressor(model, rank=32).compress()
        out_comp = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out_comp.shape, (2, 2))

    # =========================================================================
    # 9. ERCCompressor (Error-Controlled SVD)
    # =========================================================================
    def test_erc_compressor_frobenius_error_bound(self):
        """Audit ERC-SVD bounded Frobenius reconstruction error guarantee ||W - W_r||_F / ||W||_F <= eps."""
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        allowed_error = 0.10

        linear = nn.Linear(128, 128)
        compressor = ERCCompressor(model, allowed_error=allowed_error)
        svd_res = compressor.compress_linear(linear)
        svd_layer = svd_res[0] if isinstance(svd_res, tuple) else svd_res

        W_orig = linear.weight.data.float()
        W_approx = (svd_layer.U @ svd_layer.V).float()
        rel_frob_error = (torch.norm(W_orig - W_approx, p="fro") / torch.norm(W_orig, p="fro")).item()

        self.assertLessEqual(
            rel_frob_error,
            allowed_error + 1e-5,
            f"ERC-SVD violated relative Frobenius error bound: {rel_frob_error} > {allowed_error}",
        )

        comp_model, stats = compressor.compress()
        out = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out.shape, (2, 2))

    # =========================================================================
    # 10. EntropyBudgetCompressor
    # =========================================================================
    def test_entropy_budget_compressor(self):
        """Audit EntropyBudgetCompressor singular entropy global budget allocation."""
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        compressor = EntropyBudgetCompressor(model, total_budget_ratio=0.7)
        comp_model, stats = compressor.compress()
        self.assertIn("allocated_ranks", stats)
        for name, r in stats["allocated_ranks"].items():
            self.assertGreaterEqual(r, 16, f"Layer {name} rank {r} below min rank floor 16")

        out = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out.shape, (2, 2))

    # =========================================================================
    # 11. KroneckerTTLoRALayer & KroneckerTTLoRACompressor
    # =========================================================================
    def test_kronecker_tt_lora_forward_backward_and_adapt(self):
        """Audit KroneckerTTLoRALayer Kronecker product, Tensor-Train cores, LoRA bypass, and rank adaptation."""
        dense = nn.Linear(128, 128)
        layer = KroneckerTTLoRALayer.from_linear(
            base_layer=dense,
            tt_ranks=(1, 4, 4, 1),
            gamma=1.0,
            lora_r=4,
            lora_alpha=8.0,
        )
        x = torch.randn(2, 10, 128, requires_grad=True)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 128))
        self.assertFalse(torch.isnan(out).any())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.A1.grad)
        self.assertIsNotNone(layer.A2.grad)
        self.assertIsNotNone(layer.G1.grad)
        self.assertIsNotNone(layer.G2.grad)
        self.assertIsNotNone(layer.G3.grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)

        # Test TT rank adaptation
        layer.adapt_tt_ranks(delta_r=1)
        out_after_adapt = layer(x)
        self.assertEqual(out_after_adapt.shape, (2, 10, 128))
        self.assertFalse(torch.isnan(out_after_adapt).any())

        # Full model compression
        model = MockTransformerModel(hidden_size=128, num_layers=2)
        compressor = KroneckerTTLoRACompressor(model, rank=8, lora_r=4)
        comp_model, stats = compressor.compress()
        out_comp = comp_model(torch.randn(2, 8, 128))
        self.assertEqual(out_comp.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
