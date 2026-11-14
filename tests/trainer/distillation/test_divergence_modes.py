#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for LogitDistillationLoss divergence modes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.trainer.distillation import LogitDistillationLoss


class TestForwardKLDivergence:
    """Mode: forward_kl (teacher || student)."""

    def test_identical_distributions_yields_zero(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=2.0)
        t_logits = torch.randn(4, 32)
        s_logits = t_logits.clone().requires_grad_(True)
        loss = loss_fn(s_logits, t_logits)
        assert abs(loss.item()) < 1e-4

    def test_gradient_flow_and_finiteness(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=3.0)
        s_logits = torch.randn(8, 64, requires_grad=True)
        t_logits = torch.randn(8, 64)
        loss = loss_fn(s_logits, t_logits)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert torch.isfinite(s_logits.grad).all()


class TestReverseKLDivergence:
    """Mode: reverse_kl (student || teacher)."""

    def test_identical_distributions_yields_zero(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="reverse_kl", temperature=2.0)
        t_logits = torch.randn(4, 32)
        s_logits = t_logits.clone().requires_grad_(True)
        loss = loss_fn(s_logits, t_logits)
        assert abs(loss.item()) < 1e-4

    def test_gradient_flow_and_mode_seeking(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="reverse_kl", temperature=1.5)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)
        loss = loss_fn(s_logits, t_logits)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert torch.isfinite(s_logits.grad).all()


class TestBetaJSDDivergence:
    """Mode: beta_jsd (Generalized Jensen-Shannon Divergence)."""

    def test_identical_distributions_yields_zero(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="beta_jsd", temperature=2.0, beta_jsd_weight=0.5)
        t_logits = torch.randn(4, 20)
        s_logits = t_logits.clone().requires_grad_(True)
        loss = loss_fn(s_logits, t_logits)
        assert abs(loss.item()) < 1e-4

    def test_symmetry_at_half_beta(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="beta_jsd", temperature=1.0, beta_jsd_weight=0.5)
        p = torch.randn(2, 10)
        q = torch.randn(2, 10)
        loss_pq = loss_fn(p, q)
        loss_qp = loss_fn(q, p)
        assert torch.allclose(loss_pq, loss_qp, atol=1e-5)

    def test_gradient_flow(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="beta_jsd", temperature=2.0, beta_jsd_weight=0.7)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)
        loss = loss_fn(s_logits, t_logits)
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert torch.isfinite(s_logits.grad).all()


class TestDecoupledKD:
    """Mode: decoupled_kd (Decoupled Knowledge Distillation: TCKD + NCKD)."""

    def test_with_explicit_labels(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0, alpha_tckd=1.0, beta_nckd=2.0)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)
        labels = torch.randint(0, 16, (4,))
        loss = loss_fn(s_logits, t_logits, labels=labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert torch.isfinite(s_logits.grad).all()

    def test_without_labels_defaults_to_teacher_argmax(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0)
        s_logits = torch.randn(4, 16, requires_grad=True)
        t_logits = torch.randn(4, 16)
        loss = loss_fn(s_logits, t_logits, labels=None)
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None

    def test_numerical_stability_extreme_probabilities(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=1.0)
        # Extreme logits resulting in near 0 and 1 probabilities
        s_logits = torch.tensor([[-50.0, 50.0], [50.0, -50.0]], requires_grad=True)
        t_logits = torch.tensor([[50.0, -50.0], [-50.0, 50.0]])
        labels = torch.tensor([0, 1])
        loss = loss_fn(s_logits, t_logits, labels=labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(s_logits.grad).all()


class TestTopKSparseKD:
    """Mode: topk_sparse_kd (Top-K Sparse Knowledge Distillation)."""

    def test_topk_sparse_basic(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="topk_sparse_kd", temperature=2.0, top_k=4)
        s_logits = torch.randn(4, 32, requires_grad=True)
        t_logits = torch.randn(4, 32)
        loss = loss_fn(s_logits, t_logits)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert s_logits.grad is not None
        assert torch.isfinite(s_logits.grad).all()

    def test_topk_larger_than_vocab_clamped(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="topk_sparse_kd", temperature=2.0, top_k=100)
        s_logits = torch.randn(2, 10, requires_grad=True)
        t_logits = torch.randn(2, 10)
        # Should not raise IndexError
        loss = loss_fn(s_logits, t_logits)
        assert torch.isfinite(loss)

    def test_topk_k_equals_1(self):
        torch.manual_seed(42)
        loss_fn = LogitDistillationLoss(mode="topk_sparse_kd", temperature=2.0, top_k=1)
        s_logits = torch.randn(2, 8, requires_grad=True)
        t_logits = torch.randn(2, 8)
        loss = loss_fn(s_logits, t_logits)
        assert torch.isfinite(loss)


class TestInvalidDistillationMode:
    """Verifies that an unsupported distillation mode raises ValueError."""

    def test_unknown_mode_raises(self):
        loss_fn = LogitDistillationLoss(mode="unsupported_quantum_divergence")
        s = torch.randn(2, 4)
        t = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Unknown distillation mode"):
            loss_fn(s, t)
