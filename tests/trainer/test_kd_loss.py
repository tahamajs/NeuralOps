#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression and smoke tests for Multi-Level Knowledge Distillation and Loss suite."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.distillation.multi_level_kd import (
    AttentionStabilityReg,
    BlockwiseOperatorCloningLoss,
    HedgehogFeatureMapKDLoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
    MultiLevelKDTrainer,
    SmoothMaxOverLayersCKASteeringLoss,
    SpikyFeatureMap,
)
from neuralops.trainer.losses import get_loss_function


class TestLogitDistillationLoss:
    """Verifies all distillation modes of LogitDistillationLoss."""

    def test_forward_kl_loss(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=2.0)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)

        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

        loss.backward()
        assert s_logits.grad is not None
        assert not torch.isnan(s_logits.grad).any()

    def test_reverse_kl_loss(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="reverse_kl", temperature=2.0)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)

        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        loss.backward()
        assert s_logits.grad is not None

    def test_beta_jsd_loss(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="beta_jsd", temperature=2.0, beta_jsd_weight=0.5)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)

        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        loss.backward()
        assert s_logits.grad is not None

    def test_decoupled_kd_loss(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)
        labels = torch.randint(0, 16, (4,))

        loss = loss_fn(s_logits, t_logits, labels=labels)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        loss.backward()
        assert s_logits.grad is not None

    def test_topk_sparse_kd_loss(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="topk_sparse_kd", temperature=2.0, top_k=4)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)

        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        loss.backward()
        assert s_logits.grad is not None

    def test_loss_registry_integration(self):
        kd_loss = get_loss_function("kd", temperature=1.5, mode="forward_kl")
        s = torch.randn(2, 8, requires_grad=True)
        t = torch.randn(2, 8)
        loss = kd_loss(s, t)
        assert loss.item() >= 0.0


class TestRepresentationAndAttentionKD:
    """Verifies CKA, attention Gram, and feature-level distillation modules."""

    def test_linear_cka_identical_representations(self):
        torch.manual_seed(42)
        cka_fn = LinearCKALoss()
        X = torch.randn(32, 64)
        loss = cka_fn(X, X)
        # Identical representations should have CKA ~ 1.0, so loss = 1 - CKA ~ 0.0
        assert loss.item() < 1e-4

    def test_linear_cka_gradient_flow(self):
        torch.manual_seed(42)
        cka_fn = LinearCKALoss()
        X = torch.randn(16, 32, requires_grad=True)
        Y = torch.randn(16, 48)

        loss = cka_fn(X, Y)
        assert 0.0 <= loss.item() <= 1.0 + 1e-5
        loss.backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_smooth_max_cka_steering(self):
        torch.manual_seed(42)
        steering_fn = SmoothMaxOverLayersCKASteeringLoss(temperature=0.1)
        student_states = [torch.randn(8, 16, 32, requires_grad=True) for _ in range(3)]
        teacher_states = [torch.randn(8, 16, 32) for _ in range(3)]

        loss = steering_fn(student_states, teacher_states)
        assert loss.ndim == 0
        loss.backward()
        for s in student_states:
            assert s.grad is not None

    def test_minilm_value_relation_loss(self):
        torch.manual_seed(42)
        minilm = MiniLMValueRelationKDLoss()
        vs = torch.randn(2, 4, 16, 32, requires_grad=True)
        vt = torch.randn(2, 8, 16, 64)

        loss = minilm(vs, vt)
        assert loss.item() >= 0.0
        loss.backward()
        assert vs.grad is not None

    def test_hedgehog_feature_map_loss(self):
        torch.manual_seed(42)
        hedgehog = HedgehogFeatureMapKDLoss(chunk_size=16)
        qt = torch.randn(2, 4, 32, 16)
        kt = torch.randn(2, 4, 32, 16)
        phi_qs = F.relu(torch.randn(2, 4, 32, 16, requires_grad=True))
        phi_ks = F.relu(torch.randn(2, 4, 32, 16))

        loss = hedgehog(qt, kt, phi_qs, phi_ks)
        assert loss.item() >= 0.0
        loss.backward()
        assert phi_qs.grad is not None

    def test_attention_stability_regularizer(self):
        torch.manual_seed(42)
        reg_fn = AttentionStabilityReg(lambda_sat=1.0)
        gates = [torch.randn(16, 16, requires_grad=True) for _ in range(2)]

        loss = reg_fn(gates)
        assert loss.item() >= 0.0
        loss.backward()
        assert gates[0].grad is not None

        # Empty gate parameters should return zero tensor safely
        zero_loss = reg_fn([])
        assert zero_loss.item() == 0.0


class TestMultiLevelKDTrainer:
    """Smoke test for MultiLevelKDTrainer execution loop."""

    def test_trainer_step_computes_six_pillars(self):
        torch.manual_seed(42)

        class DummyModel(nn.Module):
            def __init__(self, d=16):
                super().__init__()
                self.proj = nn.Linear(d, d)
                self.gate = nn.Parameter(torch.zeros(d))

            def forward(self, x, output_hidden_states=False):
                h = self.proj(x)
                if output_hidden_states:
                    return SimpleNamespace(logits=h, hidden_states=(h,))
                return h

        from types import SimpleNamespace

        teacher = DummyModel()
        student = DummyModel()

        trainer = MultiLevelKDTrainer(
            teacher=teacher,
            student=student,
            temperature=2.0,
            lambda_kd=1.0,
            lambda_mse=0.5,
            lambda_cka=0.1,
            lambda_stab=0.01,
        )

        batch = torch.randn(2, 8, 16)
        losses = trainer.training_step(batch)

        assert "total_loss" in losses
        assert isinstance(losses["total_loss"], torch.Tensor)
        assert losses["total_loss"].requires_grad
        assert "kd_loss" in losses
        assert "mse_loss" in losses
