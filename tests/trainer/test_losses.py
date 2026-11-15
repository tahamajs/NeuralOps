#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for custom training loss functions and PyTorch reference verification."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.trainer.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    LabelSmoothingLoss,
    LinearCKALossWrapper,
    LogitKDLossWrapper,
    PolyLoss,
    get_loss_function,
)


def test_cross_entropy_loss_pytorch_parity():
    """Verify CrossEntropyLoss matches PyTorch reference exactly."""
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))

    loss_fn = CrossEntropyLoss(reduction="mean")
    loss_val = loss_fn(logits, labels)
    ref_val = F.cross_entropy(logits, labels, reduction="mean")
    assert torch.allclose(loss_val, ref_val)


def test_focal_loss_numerical_stability():
    """Verify FocalLoss is numerically stable for extreme logits."""
    # Extreme logits (very confident correct & very confident wrong)
    logits_extreme = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    labels = torch.tensor([0, 1])

    focal_fn = FocalLoss(alpha=0.5, gamma=2.0)
    loss = focal_fn(logits_extreme, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0

    # Reductions
    focal_none = FocalLoss(reduction="none")
    out_none = focal_none(logits_extreme, labels)
    assert out_none.shape == (2,)


def test_label_smoothing_loss_pytorch_parity():
    """Verify LabelSmoothingLoss matches PyTorch native label_smoothing exactly."""
    torch.manual_seed(42)
    logits = torch.randn(16, 5)
    labels = torch.randint(0, 5, (16,))
    smoothing = 0.15

    loss_fn = LabelSmoothingLoss(smoothing=smoothing)
    loss_val = loss_fn(logits, labels)
    ref_val = F.cross_entropy(logits, labels, label_smoothing=smoothing)
    assert torch.allclose(loss_val, ref_val)


def test_poly_loss_per_sample_formula():
    """Verify Poly-1 loss computes per-sample CE + eps*(1-pt)."""
    torch.manual_seed(42)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    epsilon = 1.0

    loss_fn = PolyLoss(epsilon=epsilon, reduction="mean")
    loss_val = loss_fn(logits, labels)

    # Manual per-sample calculation
    ce_per_sample = F.cross_entropy(logits, labels, reduction="none")
    pt_per_sample = torch.exp(-ce_per_sample)
    expected = (ce_per_sample + epsilon * (1.0 - pt_per_sample)).mean()
    assert torch.allclose(loss_val, expected)


def test_dice_loss():
    """Verify DiceLoss computes multi-class dice metric properly."""
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])

    dice_fn = DiceLoss(smooth=1.0)
    loss_val = dice_fn(logits, labels)
    assert 0.0 <= loss_val.item() <= 1.0


def test_contrastive_loss_dual_signatures():
    """Verify ContrastiveLoss supports both tuple and separate tensor arguments."""
    x0 = torch.randn(8, 16)
    x1 = torch.randn(8, 16)
    y = torch.randint(0, 2, (8,)).float()

    loss_fn = ContrastiveLoss(margin=1.5)
    loss1 = loss_fn((x0, x1), y)
    loss2 = loss_fn(x0, x1, y)

    assert torch.allclose(loss1, loss2)
    assert loss1.item() >= 0.0


def test_distillation_and_cka_wrappers():
    """Verify KD logit distillation and CKA loss wrappers instantiate and run."""
    student_logits = torch.randn(4, 8)
    teacher_logits = torch.randn(4, 8)

    kd_fn = LogitKDLossWrapper(temperature=2.0)
    kd_loss = kd_fn(student_logits, teacher_logits)
    assert kd_loss.dim() == 0
    assert not torch.isnan(kd_loss)

    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    cka_fn = LinearCKALossWrapper()
    cka_loss = cka_fn(x, y)
    assert cka_loss.dim() == 0
    assert not torch.isnan(cka_loss)


def test_get_loss_function_registry():
    """Verify get_loss_function retrieves registered losses and raises on unknown."""
    ce = get_loss_function("cross_entropy")
    assert isinstance(ce, CrossEntropyLoss)

    focal = get_loss_function("focal", alpha=0.5)
    assert isinstance(focal, FocalLoss)
    assert focal.alpha == 0.5

    with pytest.raises(KeyError, match="not found in LOSS_REGISTRY"):
        get_loss_function("nonexistent_loss")
