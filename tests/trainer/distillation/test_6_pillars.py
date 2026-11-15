#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for the Six Distillation Pillars."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.distillation import (
    AttentionDistillationLoss,
    CosineAlignmentLoss,
    GramMatrixLoss,
    HedgehogFeatureMapKDLoss,
    HiddenMSELoss,
    LogitDistillationLoss,
    LogitKLLoss,
    MiniLMValueRelationKDLoss,
    MultiLevelKDTrainer,
    ValueRelationLoss,
)


class TestPillar1LogitKL:
    """Pillar 1: Logit divergence matching."""

    def test_identical_logits_zero_divergence(self):
        torch.manual_seed(42)
        loss_fn = LogitKLLoss(mode="forward_kl", temperature=2.0)
        logits = torch.randn(4, 16)
        s_logits = logits.clone().requires_grad_(True)
        loss = loss_fn(s_logits, logits)
        assert loss.ndim == 0
        assert loss.item() < 1e-5

    def test_gradient_flow(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=2.0)
        s_logits = torch.randn(2, 8, requires_grad=True)
        t_logits = torch.randn(2, 8)
        loss = loss_fn(s_logits, t_logits)
        assert not torch.isnan(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert not torch.isnan(s_logits.grad).any()


class TestPillar2HiddenMSE:
    """Pillar 2: Hidden-state MSE alignment."""

    def test_identical_hidden_zero_loss(self):
        torch.manual_seed(42)
        loss_fn = HiddenMSELoss()
        h = torch.randn(2, 16, 32)
        loss = loss_fn(h, h)
        assert loss.item() == 0.0

    def test_normalized_mode(self):
        torch.manual_seed(42)
        loss_fn = HiddenMSELoss(normalize=True)
        h1 = torch.randn(2, 8, 16, requires_grad=True)
        h2 = torch.randn(2, 8, 16)
        loss = loss_fn(h1, h2)
        assert loss.item() >= 0.0
        loss.backward()
        assert h1.grad is not None

    def test_dimension_mismatch_truncation(self):
        torch.manual_seed(42)
        loss_fn = HiddenMSELoss()
        s_h = torch.randn(2, 8, 16, requires_grad=True)
        t_h = torch.randn(2, 8, 32)
        loss = loss_fn(s_h, t_h)
        assert loss.ndim == 0
        loss.backward()
        assert s_h.grad is not None


class TestPillar3AttentionParity:
    """Pillar 3: Attention distribution parity."""

    def test_identical_attention_maps(self):
        torch.manual_seed(42)
        loss_fn = AttentionDistillationLoss(temperature=1.0)
        attn = torch.randn(2, 4, 16, 16)
        loss = loss_fn(attn, attn)
        assert loss.item() < 1e-4

    def test_heterogeneous_heads_averaging(self):
        torch.manual_seed(42)
        loss_fn = AttentionDistillationLoss(temperature=1.0)
        s_attn = torch.randn(2, 4, 16, 16, requires_grad=True)
        t_attn = torch.randn(2, 8, 16, 16)
        loss = loss_fn(s_attn, t_attn)
        assert loss.ndim == 0
        loss.backward()
        assert s_attn.grad is not None

    def test_hedgehog_chunked_loss(self):
        torch.manual_seed(42)
        hedgehog = HedgehogFeatureMapKDLoss(chunk_size=16)
        qt = torch.randn(2, 4, 32, 16)
        kt = torch.randn(2, 4, 32, 16)
        phi_qs = F.relu(torch.randn(2, 4, 32, 16, requires_grad=True))
        phi_ks = F.relu(torch.randn(2, 4, 32, 16))
        loss = hedgehog(qt, kt, phi_qs, phi_ks)
        assert loss.item() >= 0.0


class TestPillar4ValueRelation:
    """Pillar 4: Value Relation Gram matrix alignment."""

    def test_identical_values(self):
        torch.manual_seed(42)
        loss_fn = ValueRelationLoss()
        v = torch.randn(2, 4, 16, 32)
        loss = loss_fn(v, v)
        assert loss.item() < 1e-4

    def test_gradient_flow_and_3d_input(self):
        torch.manual_seed(42)
        loss_fn = MiniLMValueRelationKDLoss()
        s_v = torch.randn(2, 16, 32, requires_grad=True)
        t_v = torch.randn(2, 16, 64)
        loss = loss_fn(s_v, t_v)
        assert loss.item() >= 0.0
        loss.backward()
        assert s_v.grad is not None


class TestPillar5GramMatrix:
    """Pillar 5: Gram covariance matrix alignment."""

    def test_identical_representations(self):
        torch.manual_seed(42)
        loss_fn = GramMatrixLoss()
        x = torch.randn(4, 16, 32)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-5

    def test_dimension_mismatch(self):
        torch.manual_seed(42)
        loss_fn = GramMatrixLoss()
        s = torch.randn(2, 16, 32, requires_grad=True)
        t = torch.randn(2, 16, 48)
        loss = loss_fn(s, t)
        assert loss.ndim == 0
        loss.backward()
        assert s.grad is not None


class TestPillar6CosineAlignment:
    """Pillar 6: Angular directional cosine alignment."""

    def test_identical_vectors_zero_loss(self):
        torch.manual_seed(42)
        loss_fn = CosineAlignmentLoss()
        x = torch.randn(4, 16, 32)
        loss = loss_fn(x, x)
        assert abs(loss.item()) < 1e-5

    def test_orthogonal_vectors(self):
        loss_fn = CosineAlignmentLoss()
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0]])
        loss = loss_fn(x, y)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_opposite_vectors(self):
        loss_fn = CosineAlignmentLoss()
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[-1.0, 0.0]])
        loss = loss_fn(x, y)
        assert abs(loss.item() - 2.0) < 1e-5

    def test_gradient_flow(self):
        torch.manual_seed(42)
        loss_fn = CosineAlignmentLoss()
        s = torch.randn(2, 8, 16, requires_grad=True)
        t = torch.randn(2, 8, 16)
        loss = loss_fn(s, t)
        assert not torch.isnan(loss)
        loss.backward()
        assert s.grad is not None


class TestAll6PillarsIntegration:
    """MultiLevelKDTrainer executing all 6 pillars together."""

    def test_trainer_computes_and_tracks_all_six_pillars(self):
        torch.manual_seed(42)

        class ModelWithHidden(nn.Module):
            def __init__(self, d: int = 16):
                super().__init__()
                self.proj = nn.Linear(d, d)

            def forward(self, x, output_hidden_states: bool = False):
                h = self.proj(x)
                if output_hidden_states:
                    return SimpleNamespace(logits=h, hidden_states=(h,))
                return h

        student = ModelWithHidden(16)
        teacher = ModelWithHidden(16)

        trainer = MultiLevelKDTrainer(
            teacher=teacher,
            student=student,
            temperature=2.0,
            lambda_kd=1.0,
            lambda_mse=0.5,
            lambda_attn=0.1,
            lambda_rel=0.1,
            lambda_gram=0.1,
            lambda_cos=0.1,
        )

        batch = torch.randn(4, 8, 16)
        losses = trainer.training_step(batch)

        assert "pillar1_loss" in losses
        assert "pillar2_loss" in losses
        assert "pillar3_loss" in losses
        assert "pillar4_loss" in losses
        assert "pillar5_loss" in losses
        assert "pillar6_loss" in losses
        assert "total_loss" in losses

        total = losses["total_loss"]
        assert total.requires_grad
        total.backward()
        assert student.proj.weight.grad is not None
        assert not torch.isnan(student.proj.weight.grad).any()
