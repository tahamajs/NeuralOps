#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Custom Training Losses.

Provides loss functions including distillation loss, attention matching, and regularized objectives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import Registry

logger = logging.getLogger(__name__)

LOSS_REGISTRY = Registry("loss")


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """Universal cross-entropy loss with optional fused Triton acceleration on CUDA."""

    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = -100, **kwargs: Any) -> None:
        super().__init__()
        reduction = kwargs.get("reduction", "mean")
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute module output from inputs."""
        if logits.is_cuda and self.loss.weight is None:
            try:
                from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy

                return fused_cross_entropy(
                    logits, labels, ignore_index=self.loss.ignore_index, reduction=self.loss.reduction
                )
            except (ImportError, RuntimeError, Exception) as exc:
                # allowed-swallow: triton kernel fallback to PyTorch CrossEntropyLoss
                logger.debug("Fused CE kernel unavailable, falling back to PyTorch: %s", exc)
        return self.loss(logits, labels)


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    """Numerically stable Focal Loss for addressing class imbalance."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute focal loss with numerical stability."""
        ce_loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@LOSS_REGISTRY.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with uniform label smoothing verified against PyTorch reference."""

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute label smoothing cross-entropy."""
        return F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.smoothing,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


@LOSS_REGISTRY.register("dice")
class DiceLoss(nn.Module):
    """Multi-class and binary Dice loss for classification/segmentation tasks."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute module output from inputs."""
        probs = F.softmax(logits, dim=-1)
        num_classes = probs.shape[-1]
        labels_device = labels.to(logits.device)
        one_hot = F.one_hot(labels_device, num_classes=num_classes).float()
        intersection = (probs * one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice.mean()


@LOSS_REGISTRY.register("poly")
class PolyLoss(nn.Module):
    """Poly-1 polynomial expansion loss function (Leng et al., ICLR 2022)."""

    def __init__(
        self,
        epsilon: float = 1.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute per-sample polynomial expansion loss."""
        ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        loss = ce + self.epsilon * (1.0 - pt)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@LOSS_REGISTRY.register("contrastive")
class ContrastiveLoss(nn.Module):
    """Contrastive pair margin loss function."""

    def __init__(self, margin: float = 1.0, reduction: str = "mean", eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, x0_or_pair: Any, x1_or_labels: Any, labels: Any = None) -> torch.Tensor:
        """Forward pass — compute contrastive margin loss."""
        if labels is None:
            # Called as forward((x0, x1), labels)
            x0, x1 = x0_or_pair[0], x0_or_pair[1]
            y = x1_or_labels
        else:
            # Called as forward(x0, x1, labels)
            x0 = x0_or_pair
            x1 = x1_or_labels
            y = labels

        dist = F.pairwise_distance(x0, x1, eps=self.eps)
        loss = (1.0 - y) * 0.5 * dist.pow(2) + y * 0.5 * torch.clamp(self.margin - dist, min=0.0).pow(2)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@LOSS_REGISTRY.register("kd")
@LOSS_REGISTRY.register("distillation")
@LOSS_REGISTRY.register("logit_kd")
class LogitKDLossWrapper(nn.Module):
    """Knowledge distillation logit-matching wrapper."""

    def __init__(self, temperature: float = 2.0, mode: str = "forward_kl", **kwargs: Any) -> None:
        super().__init__()
        from neuralops.trainer.distillation.multi_level_kd import LogitDistillationLoss

        self.loss_fn = LogitDistillationLoss(temperature=temperature, mode=mode, **kwargs)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.loss_fn(student_logits, teacher_logits, **kwargs)


@LOSS_REGISTRY.register("cka")
@LOSS_REGISTRY.register("linear_cka")
class LinearCKALossWrapper(nn.Module):
    """Linear Centered Kernel Alignment loss wrapper."""

    def __init__(self, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__()
        from neuralops.trainer.distillation.multi_level_kd import LinearCKALoss

        self.loss_fn = LinearCKALoss(eps=eps, **kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y)


def get_loss_function(name: str, **kwargs: Any) -> nn.Module:
    """Retrieve instantiated loss module by registered name."""
    cls = LOSS_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Loss function '{name}' not found in LOSS_REGISTRY. Available: {list(LOSS_REGISTRY.keys())}")
    return cls(**kwargs)
