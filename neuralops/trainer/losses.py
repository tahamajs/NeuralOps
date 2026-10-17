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
        if logits.is_cuda and self.loss.weight is None:
            try:
                from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy
                return fused_cross_entropy(logits, labels, ignore_index=self.loss.ignore_index, reduction=self.loss.reduction)
            except Exception:
                pass
        return self.loss(logits, labels)


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with numerically stable focal weighting."""
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        # Clamp ce_loss to prevent exp overflow/underflow
        p_t = torch.exp(-ce_loss.clamp(max=50.0))
        focal_weight = self.alpha * (1.0 - p_t).clamp(min=0.0).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


@LOSS_REGISTRY.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, **kwargs: Any) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - float(smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — compute module output with vectorization."""
        # Use PyTorch built-in label_smoothing in F.cross_entropy for maximum kernel fusion and efficiency
        if hasattr(F, "cross_entropy") and self.smoothing > 0.0:
            return F.cross_entropy(logits, labels, label_smoothing=self.smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels.to(log_probs.device)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


@LOSS_REGISTRY.register("dice")
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7, **kwargs: Any) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with safe division and stable multi-class reduction."""
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).to(probs.dtype)
        # Vectorized reduction along token dimensions
        intersection = (probs * one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice.mean()


@LOSS_REGISTRY.register("poly")
class PolyLoss(nn.Module):
    def __init__(self, epsilon: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — polynomial expansion loss with exponential clamp."""
        ce = F.cross_entropy(logits, labels)
        pt = torch.exp(-ce.clamp(max=50.0))
        return ce + self.epsilon * (1.0 - pt)


@LOSS_REGISTRY.register("contrastive")
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0, eps: float = 1e-7, **kwargs: Any) -> None:
        super().__init__()
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass — pairwise margin contrastive loss with distance stabilization."""
        dist = F.pairwise_distance(embeddings[0], embeddings[1], eps=self.eps)
        loss = (1.0 - labels) * 0.5 * dist.pow(2) + labels * 0.5 * torch.clamp(
            self.margin - dist, min=0.0
        ).pow(2)
        return loss.mean()



from neuralops.trainer.distillation.multi_level_kd import (
    HiddenAngularMSELoss,
    LatentEntropyRegularizer,
    LinearCKALoss,
    MiniLMValueRelationKDLoss,
    OrthogonalityPenaltyLoss,
    Unified6PillarDistillationLoss,
)

LOSS_REGISTRY.register("cka")(LinearCKALoss)
LOSS_REGISTRY.register("linear_cka")(LinearCKALoss)
LOSS_REGISTRY.register("value_relation")(MiniLMValueRelationKDLoss)
LOSS_REGISTRY.register("minilm_relation")(MiniLMValueRelationKDLoss)
LOSS_REGISTRY.register("angular_mse")(HiddenAngularMSELoss)
LOSS_REGISTRY.register("hidden_angular")(HiddenAngularMSELoss)
LOSS_REGISTRY.register("orthogonality")(OrthogonalityPenaltyLoss)
LOSS_REGISTRY.register("ortho_penalty")(OrthogonalityPenaltyLoss)
LOSS_REGISTRY.register("latent_entropy")(LatentEntropyRegularizer)
LOSS_REGISTRY.register("unified_6pillar")(Unified6PillarDistillationLoss)
LOSS_REGISTRY.register("six_pillar_distill")(Unified6PillarDistillationLoss)


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """Retrieve instantiated loss module by registered name."""
    cls = LOSS_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Loss function '{name}' not found in LOSS_REGISTRY. Available: {list(LOSS_REGISTRY.keys())}")
    return cls(**kwargs)


