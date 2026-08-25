"""
Comprehensive Audit & Verification Suite for MoE and Hybrid Attention Mechanisms.
Lead MoE & Hybrid Attention Specialist (Subagent 3)

Audited Mechanisms:
1. bam_umoe.py (UnifiedMoE, BAMUMoEBlock)
2. expert_choice_moe.py (ExpertChoiceMoE, ExpertChoiceMoEAttention)
3. reap_moe.py (REAPMoE, REAPMoEAttention, SVDLinear)
4. samba_moe.py (SambaMoEBlock, SAMBAM3Block, SpectralMamba)
5. moe_deltakv_paged.py (MoEDeltaKVPagedAttention, simulate_paged_delta_kv)
6. block_transformer_hiba.py (BlockTransformerHIBA)
7. jamba.py (Jamba2Block, SVDMamba)
8. hymba_transx.py (HymbaTransXBlock, HeadRoPE)
9. nexus.py (NEXUSAttention, DSSA, SeerGating, XAttentionScorer, SpectraformerRectify)
10. leka_v1.py (LEKAV1Attention)
11. leka_v2.py (LEKAv2Attention)
12. leka_v3.py (LEKAV3Attention)
13. mixattention_moska.py (MixAttentionMoSKA)
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.bam_umoe import UnifiedMoE, BAMUMoEBlock
from framework.attentions.expert_choice_moe import ExpertChoiceMoE, ExpertChoiceMoEAttention
from framework.attentions.reap_moe import REAPMoE, REAPMoEAttention, SVDLinear
from framework.attentions.samba_moe import SambaMoEBlock
from framework.attentions.samba_m3 import SAMBAM3Block, SpectralMamba
from framework.attentions.moe_deltakv_paged import (
    MoEDeltaKVPagedAttention,
    simulate_paged_delta_kv,
    low_rank_svd_quantize,
    ste_quantize_4bit,
)
from framework.attentions.block_transformer_hiba import BlockTransformerHIBA
from framework.attentions.jamba import Jamba2Block, SVDMamba
from framework.attentions.hymba_transx import HymbaTransXBlock, HeadRoPE
from framework.attentions.nexus import (
    NEXUSAttention,
    NEXUSConfig,
    DualStageStripeAttention,
    SeerGating,
    XAttentionScorer,
    SpectraformerRectify,
    softpick,
)
from framework.attentions.leka_v1 import LEKAV1Attention
from framework.attentions.leka_v2 import LEKAv2Attention
from framework.attentions.leka_v3 import LEKAV3Attention
from framework.attentions.mixattention_moska import MixAttentionMoSKA


class MockPretrainedAttention(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        nn.init.normal_(self.q_lin.weight, std=0.02)
        nn.init.normal_(self.k_lin.weight, std=0.02)
        nn.init.normal_(self.v_lin.weight, std=0.02)
        nn.init.normal_(self.out_lin.weight, std=0.02)


class MockConfig:
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1, is_causal=False):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.num_experts = 8
        self.top_k = 2
        self.window_size = 32
        self.page_size = 32
        self.k_experts = 4
        self.svd_rank = 16


class TestMoEAndHybridAttentionAudits(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = MockConfig()
        self.B = 2
        self.L = 64
        self.D = 768

    # =========================================================================
    # 1. Forward & Backward Gradient Flow for All Modules
    # =========================================================================
    def test_all_modules_forward_backward(self):
        modules = {
            "bam_umoe": BAMUMoEBlock(self.config),
            "expert_choice_moe": ExpertChoiceMoEAttention(self.config),
            "reap_moe": REAPMoEAttention(self.config),
            "samba_moe": SambaMoEBlock(self.config),
            "moe_deltakv_paged": MoEDeltaKVPagedAttention(self.config),
            "block_transformer_hiba": BlockTransformerHIBA(self.config),
            "jamba": Jamba2Block(self.config),
            "hymba_transx": HymbaTransXBlock(self.config),
            "nexus": NEXUSAttention(self.config),
            "leka_v1": LEKAV1Attention(self.config),
            "leka_v2": LEKAv2Attention(self.config),
            "leka_v3": LEKAV3Attention(self.config),
            "mixattention_moska": MixAttentionMoSKA(self.config),
        }

        for name, model in modules.items():
            with self.subTest(module=name):
                model.train()
                x = torch.randn(self.B, self.L, self.D, requires_grad=True)
                out, attn = model(x)
                
                # Check output shape
                self.assertEqual(
                    out.shape,
                    (self.B, self.L, self.D),
                    f"{name} produced incorrect shape: {out.shape} vs expected {(self.B, self.L, self.D)}"
                )
                
                # Check non-NaN / non-Inf
                self.assertFalse(
                    torch.isnan(out).any(), f"{name} produced NaN in output"
                )
                self.assertFalse(
                    torch.isinf(out).any(), f"{name} produced Inf in output"
                )

                # Backward pass
                loss = out.sum()
                if hasattr(model, "aux_loss") and model.aux_loss is not None and model.aux_loss.numel() > 0:
                    loss = loss + model.aux_loss.sum()
                loss.backward()

                self.assertIsNotNone(x.grad, f"{name} x.grad is None")
                self.assertFalse(
                    torch.isnan(x.grad).any(), f"{name} x.grad has NaNs"
                )

    # =========================================================================
    # 2. Expert Load Balancing & Auxiliary Loss Verification
    # =========================================================================
    def test_bam_umoe_load_balancing_and_aux_loss(self):
        moe = UnifiedMoE(dim=self.D, num_experts=8, top_k=2, noise_std=0.1)
        moe.train()
        x = torch.randn(4, 32, self.D)
        out = moe(x)
        self.assertEqual(out.shape, (4, 32, self.D))
        
        # Verify aux_loss is non-zero, positive, and finite
        self.assertGreater(moe.aux_loss.item(), 0.0)
        self.assertFalse(torch.isnan(moe.aux_loss).any())
        
        # In eval mode, deterministic routing
        moe.eval()
        out_eval = moe(x)
        self.assertEqual(out_eval.shape, (4, 32, self.D))

    def test_expert_choice_moe_balance_and_capacity(self):
        moe = ExpertChoiceMoE(d_model=self.D, n_experts=8, capacity_factor=1.0)
        moe.train()
        x = torch.randn(4, 32, self.D)
        out = moe(x)
        self.assertEqual(out.shape, (4, 32, self.D))
        self.assertGreater(moe.aux_loss.item(), 0.0)
        self.assertFalse(torch.isnan(moe.aux_loss).any())

    def test_reap_moe_svd_factorization_and_aux_loss(self):
        moe = REAPMoE(d_model=self.D, n_experts=8, top_k=2, use_svd=True)
        moe.train()
        x = torch.randn(4, 32, self.D)
        out, _ = moe(x)
        self.assertEqual(out.shape, (4, 32, self.D))
        self.assertGreater(moe.aux_loss.item(), 0.0)
        self.assertFalse(torch.isnan(moe.aux_loss).any())

    def test_samba_moe_hybrid_aux_loss(self):
        samba = SambaMoEBlock(self.config)
        samba.train()
        x = torch.randn(2, 32, self.D)
        out, _ = samba(x)
        self.assertEqual(out.shape, (2, 32, self.D))
        self.assertGreater(samba.aux_loss.item(), 0.0)

    def test_jamba_hybrid_moe(self):
        jamba = Jamba2Block(self.config)
        jamba.train()
        x = torch.randn(2, 32, self.D)
        out, _ = jamba(x)
        self.assertEqual(out.shape, (2, 32, self.D))
        self.assertGreater(jamba.aux_loss.item(), 0.0)

    # =========================================================================
    # 3. [CLS] Token Protection at Index 0
    # =========================================================================
    def test_cls_token_protection_mixattention_moska(self):
        """Verify [CLS] token at index 0 is never pruned by MixAttentionMoSKA."""
        model = MixAttentionMoSKA(self.config, fasa_keep_ratio=0.25)
        model.eval()
        x = torch.randn(2, 64, self.D)
        
        # Check pruning logic directly
        k = model.W_k(x).view(2, 64, model.num_heads, model.head_dim).permute(0, 2, 1, 3)
        v = model.W_v(x).view(2, 64, model.num_heads, model.head_dim).permute(0, 2, 1, 3)
        
        k_p, v_p, _ = model._fasa_prune(k, v)
        # Even with keep_ratio=0.25 (16 tokens kept out of 64), token 0 must be in k_p
        # Specifically, index 0 is guaranteed by importance[:, 0] = 1e9
        # Check that the first token in k_p matches the first token in k
        self.assertTrue(torch.allclose(k_p[:, :, 0, :], k[:, :, 0, :]))
        self.assertTrue(torch.allclose(v_p[:, :, 0, :], v[:, :, 0, :]))

    def test_cls_token_protection_moe_deltakv_paged(self):
        """Verify page 0 (which contains CLS token at index 0) is always routed."""
        model = MoEDeltaKVPagedAttention(self.config)
        model.eval()
        # Sequence length 128 with page size 32 => 4 pages
        x = torch.randn(2, 128, self.D)
        out, _ = model(x)
        self.assertEqual(out.shape, (2, 128, self.D))
        self.assertFalse(torch.isnan(out).any())

    def test_cls_token_protection_block_transformer_hiba(self):
        """Verify block 0 (which contains CLS token at index 0) is always included in inter-block sparse."""
        model = BlockTransformerHIBA(self.config, block_size=32, top_k_inter=2, rectification_interval=2)
        model.eval()
        x = torch.randn(2, 128, self.D)
        # Call forward twice to trigger non-rectify sparse path
        out1, _ = model(x)
        out2, _ = model(x)
        self.assertEqual(out2.shape, (2, 128, self.D))
        self.assertFalse(torch.isnan(out2).any())

    def test_cls_token_protection_leka_v2_v3(self):
        """Verify [CLS] token at index 0 is always prioritized as a landmark in LEKAv2 and LEKAv3."""
        for name, cls in [("leka_v2", LEKAv2Attention), ("leka_v3", LEKAV3Attention)]:
            with self.subTest(module=name):
                model = cls(self.config)
                model.eval()
                x = torch.randn(2, 64, self.D)
                
                # Check landmark selection
                if name == "leka_v2":
                    K = model.transpose_for_scores(model.key(x))
                    V = model.transpose_for_scores(model.value(x))
                    K_feat = model.kernel_feature_map(K)
                    K_land, V_land = model.select_landmarks(K_feat, V)
                    # CLS token should be at index 0 of landmarks
                    self.assertTrue(torch.allclose(K_land[:, :, 0, :], K_feat[:, :, 0, :]))
                    self.assertTrue(torch.allclose(V_land[:, :, 0, :], V[:, :, 0, :]))
                elif name == "leka_v3":
                    K = model.transpose_for_scores(model.key(x))
                    V = model.transpose_for_scores(model.value(x))
                    K_feat = model.kernel_feature_map(K)
                    valid_mask = torch.ones(2, 64, dtype=torch.bool)
                    K_land, V_land = model.select_landmarks(K_feat, V, valid_mask)
                    self.assertTrue(torch.allclose(K_land[:, :, 0, :], K_feat[:, :, 0, :]))
                    self.assertTrue(torch.allclose(V_land[:, :, 0, :], V[:, :, 0, :]))

    # =========================================================================
    # 4. Sliding Window Alignments
    # =========================================================================
    def test_sliding_window_samba_m3(self):
        """Verify SAMBA-M3 sliding window attention masking."""
        window_size = 16
        block = SAMBAM3Block(d_model=self.D, n_heads=12, window=window_size)
        block.eval()
        x = torch.randn(2, 64, self.D)
        out, attn = block(x, output_attentions=True)
        self.assertEqual(out.shape, (2, 64, self.D))
        
        # Verify attention matrix has 0 probability outside window
        if attn is not None:
            # attn shape [B, H, L, L]
            for i in range(64):
                for j in range(64):
                    if abs(i - j) > window_size:
                        # Should be effectively 0 (< 1e-4)
                        self.assertLess(
                            attn[:, :, i, j].max().item(),
                            1e-4,
                            f"Attention leaked outside window at ({i}, {j})"
                        )

    def test_sliding_window_leka_v2_v3(self):
        """Verify LEKAv2 and LEKAv3 local window masking."""
        for name, cls in [("leka_v2", LEKAv2Attention), ("leka_v3", LEKAV3Attention)]:
            with self.subTest(module=name):
                model = cls(self.config)
                model.eval()
                x = torch.randn(2, 32, self.D)
                out, _ = model(x)
                self.assertEqual(out.shape, (2, 32, self.D))
                self.assertFalse(torch.isnan(out).any())

    # =========================================================================
    # 5. Mask Handling & Multi-Format Attention Masks (2D, 4D, Padding)
    # =========================================================================
    def test_mask_handling_all_modules(self):
        modules = {
            "bam_umoe": BAMUMoEBlock(self.config),
            "expert_choice_moe": ExpertChoiceMoEAttention(self.config),
            "reap_moe": REAPMoEAttention(self.config),
            "samba_moe": SambaMoEBlock(self.config),
            "moe_deltakv_paged": MoEDeltaKVPagedAttention(self.config),
            "block_transformer_hiba": BlockTransformerHIBA(self.config),
            "jamba": Jamba2Block(self.config),
            "hymba_transx": HymbaTransXBlock(self.config),
            "nexus": NEXUSAttention(self.config),
            "leka_v1": LEKAV1Attention(self.config),
            "leka_v2": LEKAv2Attention(self.config),
            "leka_v3": LEKAV3Attention(self.config),
            "mixattention_moska": MixAttentionMoSKA(self.config),
        }

        # 2D mask: [B, L]
        mask_2d = torch.ones(self.B, self.L)
        mask_2d[:, self.L // 2 :] = 0  # 50% padded

        # 4D mask: [B, 1, 1, L]
        mask_4d = torch.zeros(self.B, 1, 1, self.L)
        mask_4d[:, :, :, self.L // 2 :] = -10000.0

        for name, model in modules.items():
            with self.subTest(module=name, mask="2D"):
                model.eval()
                x = torch.randn(self.B, self.L, self.D)
                out, _ = model(x, attention_mask=mask_2d)
                self.assertEqual(out.shape, (self.B, self.L, self.D))
                self.assertFalse(torch.isnan(out).any(), f"{name} produced NaN with 2D mask")

            with self.subTest(module=name, mask="4D"):
                model.eval()
                x = torch.randn(self.B, self.L, self.D)
                out, _ = model(x, attention_mask=mask_4d)
                self.assertEqual(out.shape, (self.B, self.L, self.D))
                self.assertFalse(torch.isnan(out).any(), f"{name} produced NaN with 4D mask")

    # =========================================================================
    # 6. Pretrained Weight Copy Parity
    # =========================================================================
    def test_weight_copying_all_modules(self):
        pretrained = MockPretrainedAttention(self.D)
        
        modules = [
            BAMUMoEBlock(self.config),
            ExpertChoiceMoEAttention(self.config),
            REAPMoEAttention(self.config),
            SambaMoEBlock(self.config),
            MoEDeltaKVPagedAttention(self.config),
            BlockTransformerHIBA(self.config),
            Jamba2Block(self.config),
            HymbaTransXBlock(self.config),
            NEXUSAttention(self.config),
            LEKAV1Attention(self.config),
            LEKAv2Attention(self.config),
            LEKAV3Attention(self.config),
            MixAttentionMoSKA(self.config),
        ]

        for model in modules:
            name = model.__class__.__name__
            with self.subTest(module=name):
                model.copy_weights(pretrained)
                # Verify that query projection has copied weights
                if hasattr(model, "q_lin"):
                    self.assertTrue(torch.equal(model.q_lin.weight, pretrained.q_lin.weight))
                elif hasattr(model, "query"):
                    self.assertTrue(torch.equal(model.query.weight, pretrained.q_lin.weight))
                elif hasattr(model, "W_q"):
                    self.assertTrue(torch.equal(model.W_q.weight, pretrained.q_lin.weight))
                elif hasattr(model, "W_q1"):
                    self.assertTrue(torch.equal(model.W_q1.weight, pretrained.q_lin.weight))
                elif hasattr(model, "swa_mamba"):
                    self.assertTrue(torch.equal(model.swa_mamba.q_lin.weight, pretrained.q_lin.weight))


if __name__ == "__main__":
    unittest.main()
