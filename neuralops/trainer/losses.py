from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.registry import Registry

LOSS_REGISTRY = Registry("loss")


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, **kwargs) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        return self.loss(logits, labels)


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce_loss)
        return (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()


@LOSS_REGISTRY.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, **kwargs) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels.to(log_probs.device)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


@LOSS_REGISTRY.register("dice")
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **kwargs) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
        intersection = (probs * one_hot).sum(dim=0)
        union = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


@LOSS_REGISTRY.register("poly")
class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, **kwargs) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        ce = F.cross_entropy(logits, labels)
        pt = torch.exp(-ce)
        return ce + self.epsilon * (1 - pt)


@LOSS_REGISTRY.register("contrastive")
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, **kwargs) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels) -> Any:
        """Forward pass — compute module output from inputs."""
        dist = F.pairwise_distance(embeddings[0], embeddings[1])
        loss = (1 - labels) * 0.5 * dist.pow(2) + labels * 0.5 * torch.clamp(
            self.margin - dist, min=0
        ).pow(2)
        return loss.mean()
