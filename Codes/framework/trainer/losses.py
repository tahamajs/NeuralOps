import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import Registry

LOSS_REGISTRY = Registry("loss")

@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, labels):
        return self.loss(logits, labels)

@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

@LOSS_REGISTRY.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1))
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()