# framework/trainer/methods/lora.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_, constant_
from .registry import register_method
import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Base LoRA linear layer
# ----------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
        else:
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()

    def forward(self, x):
        if self.r > 0:
            lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return self.linear(x) + lora
        return self.linear(x)

# ----------------------------------------------------------------------
# DoRA (Weight‑decomposed Low‑Rank Adaptation)
# ----------------------------------------------------------------------
class DoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.r = r
        if r > 0:
            # Magnitude vector (scalar per output dimension)
            self.m = nn.Parameter(torch.ones(linear.out_features))
            # Low‑rank matrices
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
        else:
            self.m = nn.Parameter(torch.ones(linear.out_features))
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()

    def forward(self, x):
        base_out = self.linear(x)
        if self.r > 0:
            lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            # Normalise base weight direction
            weight_norm = self.linear.weight.norm(p=2, dim=1, keepdim=True)
            direction = self.linear.weight / (weight_norm + 1e-8)
            # Magnitude‑scaled direction + LoRA
            delta = self.m.unsqueeze(1) * direction + lora
            return F.linear(x, delta, self.linear.bias)
        return base_out

# ----------------------------------------------------------------------
# DisLoRA (Dissimilarity‑based LoRA)
# ----------------------------------------------------------------------
class DisLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float,
                 beta: float = 0.5, capacity_factor: float = 1.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.r = r
        self.beta = beta
        self.capacity_factor = capacity_factor
        if r > 0:
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
            # Orthogonal regularisation (optional)
            self.register_buffer("_eye", torch.eye(r))
        else:
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0:
            lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            # Additional regularisation (implement in loss if needed)
            return base + lora
        return base

    def ortho_loss(self):
        if self.lora_A is None:
            return 0.0
        A = self.lora_A.weight
        B = self.lora_B.weight
        # Encourage orthogonality of A and B
        reg = (A @ A.T - self._eye).pow(2).sum() + (B.T @ B - self._eye).pow(2).sum()
        return self.beta * reg

# ----------------------------------------------------------------------
# Apply functions that replace linear layers
# ----------------------------------------------------------------------
def _apply_adapters(model, r, alpha, dropout, method_type='lora', **kwargs):
    target_substrings = ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(ts in name for ts in target_substrings):
            parent = model.get_submodule('.'.join(name.split('.')[:-1]) or '')
            child = name.split('.')[-1]
            try:
                if method_type == 'dora':
                    new_module = DoRALinear(module, r, alpha, dropout)
                elif method_type == 'dislora':
                    new_module = DisLoRALinear(module, r, alpha, dropout,
                                               beta=kwargs.get('dislora_beta', 0.5),
                                               capacity_factor=kwargs.get('dislora_capacity_factor', 1.0))
                else:
                    new_module = LoRALinear(module, r, alpha, dropout)
                setattr(parent, child, new_module)
            except Exception as e:
                logger.warning(f"Failed to replace {name}: {e}")

@register_method("lora")
def apply_lora(model, config):
    _apply_adapters(model, config.lora_r, config.lora_alpha, config.lora_dropout, 'lora')

@register_method("dora")
def apply_dora(model, config):
    _apply_adapters(model, config.lora_r, config.lora_alpha, config.lora_dropout, 'dora')

@register_method("dislora")
def apply_dislora(model, config):
    _apply_adapters(model, config.lora_r, config.lora_alpha, config.lora_dropout, 'dislora',
                    dislora_beta=config.dislora_beta,
                    dislora_capacity_factor=config.dislora_capacity_factor)