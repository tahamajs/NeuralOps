"""LoRA, DoRA, DisLoRA adapters."""
import math, torch.nn as nn, torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_
from .registry import register_method
import logging

logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    # ... (same as legacy, but streamlined)
    def __init__(self, linear, r, alpha, dropout):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None: linear.bias.requires_grad = False
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

# DoRALinear and DisLoRALinear similarly (omitted for brevity – full code in final package)

def _apply_adapters(model, r, alpha, dropout, method_type='lora', **kwargs):
    targets = ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or "classifier" in n or "pooler" in n:
            continue
        if any(t in n for t in targets):
            parent = model.get_submodule('.'.join(n.split('.')[:-1]) or '')
            child = n.split('.')[-1]
            try:
                if method_type == 'dora':
                    setattr(parent, child, DoRALinear(m, r, alpha, dropout))
                elif method_type == 'dislora':
                    setattr(parent, child, DisLoRALinear(m, r, alpha, dropout,
                                                         beta=kwargs.get('dislora_beta', 0.5),
                                                         capacity_factor=kwargs.get('dislora_capacity_factor', 1.0)))
                else:
                    setattr(parent, child, LoRALinear(m, r, alpha, dropout))
            except Exception as e:
                logger.warning(f"Failed {n}: {e}")

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