"""
Mixout: Regularisation by mixing weights with a frozen copy.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_method

logger = logging.getLogger(__name__)


class MixoutLinear(nn.Module):
    def __init__(self, linear: nn.Linear, p: float) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.p = p
        self.register_buffer("target_weight", self.weight.data.clone())
        if self.bias is not None:
            self.register_buffer("target_bias", self.bias.data.clone())

    def forward(self, x) -> Any:
        """Forward pass — compute module output from inputs."""
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.p))
            mixed_weight = self.weight * mask + self.target_weight * (1 - mask)
            if self.bias is not None:
                mask_bias = torch.bernoulli(torch.full_like(self.bias, 1 - self.p))
                mixed_bias = self.bias * mask_bias + self.target_bias * (1 - mask_bias)
            else:
                mixed_bias = None
            return F.linear(x, mixed_weight, mixed_bias)
        return F.linear(x, self.weight, self.bias)


def replace_linears_with_mixout(
    model: nn.Module, p: float, skip_classifier: bool = True
) -> None:
    """Replace linears with mixout."""
    for name, module in list(model.named_children()):
        if skip_classifier and "classifier" in name:
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, MixoutLinear(module, p))
        else:
            replace_linears_with_mixout(module, p, skip_classifier)


@register_method("mixout")
def apply_mixout(model, config) -> None:
    """Apply mixout."""
    replace_linears_with_mixout(model, config.mixout_p)
    logger.info("Mixout applied.")
