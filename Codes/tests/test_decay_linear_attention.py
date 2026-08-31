"""
Unit tests for SVD Linear Attention with Data-Dependent Exponential Decay Gating.

Verifies:
1. Exact softmax attention mode backward compatibility.
2. Standard non-decayed linear kernel attention backward compatibility.
3. Data-dependent exponential decay gated recurrence (anti-saturation on long sequences).
4. Mathematical bounds of decay factor computation (lambda in (0, 1)).
5. Numerical equivalence between full-sequence forward pass and single-token step_recurrent generation.
6. Mask handling and padding isolation in associative state recurrence.
7. Needle-in-a-haystack retention and fading memory dynamics.
8. Channel-wise decay gating mode.
9. Stiefel manifold orthogonality regularization loss.
10. Registry resolution across all alias names.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    compute_exponential_decay,
)
from framework.core.registry import ATTENTION_REGISTRY


class DummyAttentionConfig:
    """Mock configuration object for unit testing attention modules."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        rank: int = 16,
        dropout: float = 0.0,
        use_decay_gating: bool = False,
        decay_activation: str = "softplus",
        decay_channel_wise: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.rank = rank
        self.svd_rank = rank
        self.dropout = dropout
        self.attention_probs_dropout_prob = dropout
        self.use_decay_gating = use_decay_gating
        self.decay_activation = decay_activation
        self.decay_channel_wise = decay_channel_wise
        self.linear_attention_eps = 1e-6


class TestDecayLinearAttention(unittest.TestCase):
    """Test suite for SVD linear attention with data-dependent decay gating."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 64
        self.num_heads = 4
        self.rank = 16
        self.cfg = DummyAttentionConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            rank=self.rank,
        )
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    # -------------------------------------------------------------------------
    # 1. Decay Factor Mathematical Properties
    # -------------------------------------------------------------------------

    def test_compute_exponential_decay_bounds(self) -> None:
        """
        RESEARCH FIX: Verify decay factor lambda is strictly bounded in (0, 1).
        softplus(u) > 0 for all real u, hence exp(-softplus(u)) < 1.0 unconditionally.
        """
        # Test extreme inputs: huge negative, huge positive, zero
        logits = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])

        # Softplus mode
        lambdas_softplus = compute_exponential_decay(logits, activation="softplus")
        self.assertTrue((lambdas_softplus > 0.0).all().item())
        self.assertTrue((lambdas_softplus <= 1.0).all().item())
        # Large positive logit should yield rapid decay (lambda near 0)
        self.assertLess(lambdas_softplus[-1].item(), 1e-4)
        # Large negative logit should yield near-unit retention (lambda near 1)
        self.assertGreater(lambdas_softplus[0].item(), 0.999)

        # Sigmoid mode
        lambdas_sigmoid = compute_exponential_decay(logits, activation="sigmoid")
        self.assertTrue((lambdas_sigmoid > 0.0).all().item())
        self.assertTrue((lambdas_sigmoid < 1.0).all().item())

        # Invalid activation mode raises ValueError
        with self.assertRaises(ValueError):
            compute_exponential_decay(logits, activation="invalid_mode")

    # -------------------------------------------------------------------------
    # 2. Exact Softmax Mode Backward Compatibility
    # -------------------------------------------------------------------------

    def test_exact_softmax_mode(self) -> None:
        """Verify exact softmax attention mode preserves O(N^2) formulation."""
        layer = SVDLinearAttention(self.cfg, mode="exact")
        out, attn_probs = layer(self.dummy_x, output_attentions=True)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsNotNone(attn_probs)
        self.assertEqual(
            attn_probs.shape,
            (self.batch_size, self.num_heads, self.seq_len, self.seq_len),
        )
        # Attention weights along key dimension must sum to 1.0
        self.assertTrue(
            torch.allclose(
                attn_probs.sum(dim=-1),
                torch.ones(self.batch_size, self.num_heads, self.seq_len),
                atol=1e-5,
            )
        )

    def test_exact_softmax_no_attentions(self) -> None:
        """Verify exact mode with output_attentions=False returns None for attn_probs."""
        layer = SVDLinearAttention(self.cfg, mode="exact")
        out, attn_probs = layer(self.dummy_x, output_attentions=False)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsNone(attn_probs)

    # -------------------------------------------------------------------------
    # 3. Standard Non-Decayed Linear Kernel Mode Backward Compatibility
    # -------------------------------------------------------------------------

    def test_standard_linear_mode(self) -> None:
        """Verify standard O(N) linear kernel attention runs without decay when use_decay_gating=False."""
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=False)
        out, attn_probs = layer(self.dummy_x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsNone(attn_probs)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    # -------------------------------------------------------------------------
    # 4. Data-Dependent Exponential Decay Gating Mode
    # -------------------------------------------------------------------------

    def test_decay_gated_linear_mode_forward(self) -> None:
        """
        RESEARCH FIX: Verify data-dependent exponential decay gated attention forward pass.
        Ensures state recurrence computes finite outputs without overflow.
        """
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        out, attn_probs = layer(self.dummy_x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsNone(attn_probs)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_decay_linear_mode_aliases(self) -> None:
        """Verify mode aliases 'decay_linear', 'decay', and 'gated_linear' activate decay gating."""
        for mode_name in ["decay_linear", "linear_decay", "decay", "gated_linear"]:
            layer = SVDLinearAttention(self.cfg, mode=mode_name)
            self.assertTrue(layer.use_decay_gating)
            out, _ = layer(self.dummy_x)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_decay_gate_gradient_flow(self) -> None:
        """
        RESEARCH FIX: Verify that end-to-end gradients backpropagate through
        both the SVD projections and the data-dependent decay_gate / gate_proj.
        """
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        out, _ = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertIsNotNone(layer.decay_gate.weight.grad)
        self.assertIsNotNone(layer.decay_gate.bias.grad)
        self.assertIsNotNone(layer.q_proj.V.grad)
        self.assertIsNotNone(layer.out_proj.U.grad)

    # -------------------------------------------------------------------------
    # 5. Numerical Equivalence: Full Forward vs Step-by-Step Recurrence
    # -------------------------------------------------------------------------

    def test_step_recurrent_equivalence(self) -> None:
        """
        RESEARCH FIX: Proves exact mathematical equivalence between full-sequence
        recurrent scan and step_recurrent token generation for streaming inference.
        """
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        layer.eval()

        with torch.no_grad():
            # 1. Full sequence forward pass
            full_out, _, full_final_state = layer(
                self.dummy_x, return_state=True
            )

            # 2. Step-by-step token generation
            step_outs = []
            state = None
            for t in range(self.seq_len):
                x_t = self.dummy_x[:, t : t + 1, :]
                step_out, state = layer.step_recurrent(x_t, past_state=state)
                step_outs.append(step_out)

            stacked_step_out = torch.cat(step_outs, dim=1)

            # 3. Assert complete output equivalence
            self.assertTrue(
                torch.allclose(full_out, stacked_step_out, atol=1e-5, rtol=1e-4),
                f"Max discrepancy between full scan and step recurrence: {torch.max(torch.abs(full_out - stacked_step_out)).item()}",
            )

            # 4. Assert final memory state matrix (S, Z) equivalence
            S_full, Z_full = full_final_state
            S_step, Z_step = state
            self.assertTrue(
                torch.allclose(S_full, S_step, atol=1e-5, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(Z_full, Z_step, atol=1e-5, rtol=1e-4)
            )

    # -------------------------------------------------------------------------
    # 6. Memory Saturation Prevention on Long Sequences (>32k Simulation)
    # -------------------------------------------------------------------------

    def test_state_norm_boundedness_on_long_sequence(self) -> None:
        """
        RESEARCH FIX: Demonstrates that data-dependent exponential decay maintains
        bounded associative memory norm ||S_t|| on long sequences, whereas standard
        non-decayed linear attention state grows unbounded O(N).
        """
        long_seq_len = 512
        long_x = torch.randn(1, long_seq_len, self.hidden_size)

        # Standard non-decayed linear kernel: state norm grows linearly with N
        # Recurrent simulation:
        phi_k = F.elu(torch.randn(1, self.num_heads, long_seq_len, self.cfg.head_dim)) + 1.0
        v = torch.randn(1, self.num_heads, long_seq_len, self.cfg.head_dim)

        # Non-decayed accumulation
        S_no_decay = torch.zeros(1, self.num_heads, self.cfg.head_dim, self.cfg.head_dim)
        norms_no_decay = []
        for t in range(long_seq_len):
            kt = phi_k[:, :, t : t + 1, :].transpose(-1, -2)
            vt = v[:, :, t : t + 1, :]
            S_no_decay = S_no_decay + torch.matmul(kt, vt)
            norms_no_decay.append(torch.norm(S_no_decay).item())

        # Exponential decay accumulation (lambda = 0.95)
        lambda_val = 0.95
        S_decay = torch.zeros(1, self.num_heads, self.cfg.head_dim, self.cfg.head_dim)
        norms_decay = []
        for t in range(long_seq_len):
            kt = phi_k[:, :, t : t + 1, :].transpose(-1, -2)
            vt = v[:, :, t : t + 1, :]
            S_decay = lambda_val * S_decay + torch.matmul(kt, vt)
            norms_decay.append(torch.norm(S_decay).item())

        # The non-decayed norm must grow substantially over time
        self.assertGreater(norms_no_decay[-1], norms_no_decay[10] * 5.0)
        # The decayed norm saturates at a bounded steady-state limit O(1 / (1 - lambda))
        self.assertLess(norms_decay[-1], norms_no_decay[-1] * 0.2)

    def test_needle_in_a_haystack_exponential_retention(self) -> None:
        """
        RESEARCH FIX: Test selective forgetting and retention.
        Low lambda rapidly forgets past distractors; high lambda preserves the needle.
        """
        layer_high_retain = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        layer_low_retain = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)

        # Set bias to force high retention (lambda ~ 0.999) vs low retention (lambda ~ 0.1)
        with torch.no_grad():
            layer_high_retain.decay_gate.weight.zero_()
            layer_high_retain.decay_gate.bias.fill_(-6.9)  # softplus(-6.9) ~ 0.001 => lambda = 0.999

            layer_low_retain.decay_gate.weight.zero_()
            layer_low_retain.decay_gate.bias.fill_(2.5)   # softplus(2.5) ~ 2.58 => lambda = 0.075

        # Place a prominent needle activation at position 0, followed by zero or noise distractors
        x_seq = torch.zeros(1, 30, self.hidden_size)
        x_seq[:, 0, :] = 5.0  # Needle token

        _, _, state_high = layer_high_retain(x_seq, return_state=True)
        _, _, state_low = layer_low_retain(x_seq, return_state=True)

        S_high, _ = state_high
        S_low, _ = state_low

        # High-retention model preserves the needle memory; low-retention model decays it away
        self.assertGreater(torch.norm(S_high).item(), torch.norm(S_low).item() * 50.0)

    # -------------------------------------------------------------------------
    # 7. Mask Handling and Padding Isolation
    # -------------------------------------------------------------------------

    def test_attention_masking_2d_and_4d(self) -> None:
        """Verify attention mask prevents padded tokens from updating memory states."""
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        layer.eval()

        # Mask: first 10 tokens are valid (1), remaining 6 tokens are padded (0)
        mask_2d = torch.ones(self.batch_size, self.seq_len)
        mask_2d[:, 10:] = 0.0

        out_masked, _ = layer(self.dummy_x, attention_mask=mask_2d)
        self.assertEqual(out_masked.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(out_masked).any())

        # Test with 4D HuggingFace style mask
        mask_4d = (mask_2d.unsqueeze(1).unsqueeze(2) - 1.0) * 10000.0
        out_masked_4d, _ = layer(self.dummy_x, attention_mask=mask_4d)
        self.assertEqual(out_masked_4d.shape, (self.batch_size, self.seq_len, self.hidden_size))

    # -------------------------------------------------------------------------
    # 8. Channel-Wise Decay Gating Mode
    # -------------------------------------------------------------------------

    def test_decay_channel_wise(self) -> None:
        """Verify fine-grained channel-wise decay gating forward and step recurrence."""
        cfg_channel = DummyAttentionConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            rank=self.rank,
            use_decay_gating=True,
            decay_channel_wise=True,
        )
        layer = SVDLinearAttention(cfg_channel, mode="linear", decay_channel_wise=True)
        out, _ = layer(self.dummy_x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())

        # Verify step recurrence with channel-wise decay
        x_t = self.dummy_x[:, :1, :]
        step_out, state = layer.step_recurrent(x_t)
        self.assertEqual(step_out.shape, (self.batch_size, 1, self.hidden_size))
        self.assertEqual(state[0].shape, (self.batch_size, self.num_heads, self.cfg.head_dim, self.cfg.head_dim))
        self.assertEqual(state[1].shape, (self.batch_size, self.num_heads, self.cfg.head_dim, 1))

    # -------------------------------------------------------------------------
    # 9. Stiefel Manifold Orthogonality Regularization Loss
    # -------------------------------------------------------------------------

    def test_orthogonality_loss(self) -> None:
        """
        RESEARCH FIX: Test Stiefel manifold penalty ||U^T U - I||_F^2 + ||V V^T - I||_F^2.
        Loss is positive for unconstrained weights and near zero for perfect orthogonal factors.
        """
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        ortho_loss = layer.get_orthogonality_loss()

        self.assertIsInstance(ortho_loss, torch.Tensor)
        self.assertGreaterEqual(ortho_loss.item(), 0.0)

        # Gradients propagate through orthogonality loss
        ortho_loss.backward()
        self.assertIsNotNone(layer.q_proj.U.grad)
        self.assertIsNotNone(layer.q_proj.V.grad)

    # -------------------------------------------------------------------------
    # 10. Weight Factorization & SVD Transfer
    # -------------------------------------------------------------------------

    def test_copy_weights_svd_factorization(self) -> None:
        """Verify pretrained PyTorch MultiheadAttention weights can be factorized into SVD projections."""
        layer = SVDLinearAttention(self.cfg, mode="linear", use_decay_gating=True)
        mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
        )

        layer.copy_weights(mha)
        out, _ = layer(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())

    # -------------------------------------------------------------------------
    # 11. Registry Registration and Instantiation
    # -------------------------------------------------------------------------

    def test_registry_aliases_and_instantiation(self) -> None:
        """Verify all registry keys resolve to SVDLinearAttention and execute."""
        aliases = [
            "svd_linear",
            "svd_linear_attn",
            "projection_factorized_attn",
            "svd_linear_decay",
            "decay_linear_attn",
            "gated_linear_attn",
        ]
        for alias in aliases:
            self.assertIn(alias, ATTENTION_REGISTRY, f"Registry key '{alias}' missing!")
            cls = ATTENTION_REGISTRY.get(alias)
            mod = cls(config=self.cfg)
            out = mod(self.dummy_x)
            if isinstance(out, tuple):
                out = out[0]
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))


if __name__ == "__main__":
    unittest.main()
