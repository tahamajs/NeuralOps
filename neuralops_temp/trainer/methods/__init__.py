"""
Training method plugins: LoRA, DoRA, DisLoRA, Fisher, GeLoRA, LLRD, Mixout, R‑Drop, R3F.
All methods are registered via the METHOD_REGISTRY and can be applied by name.
"""

from .fisher import apply_fisher_lora, compute_fisher_importance
from .gelora import (
    apply_fisher_gelora,
    collect_hidden_states,
    compute_gelora_base_ranks,
    estimate_intrinsic_dimension_pytorch,
)
from .llrd import apply_llrd, get_llrd_param_groups
from .lora import (
    DisLoRALinear,
    DoRALinear,
    LoRALinear,
    apply_dislora,
    apply_dora,
    apply_lora,
)
from .mixout import MixoutLinear, apply_mixout
from .rdrop_r3f import MethodTrainer, apply_r3f, apply_rdrop, r3f_forward, rdrop_loss
from .registry import METHOD_REGISTRY, available_methods, get_method, register_method

__all__ = [
    "METHOD_REGISTRY",
    "register_method",
    "get_method",
    "available_methods",
    "apply_lora",
    "apply_dora",
    "apply_dislora",
    "LoRALinear",
    "DoRALinear",
    "DisLoRALinear",
    "compute_fisher_importance",
    "apply_fisher_lora",
    "estimate_intrinsic_dimension_pytorch",
    "collect_hidden_states",
    "compute_gelora_base_ranks",
    "apply_fisher_gelora",
    "get_llrd_param_groups",
    "apply_llrd",
    "MixoutLinear",
    "apply_mixout",
    "rdrop_loss",
    "r3f_forward",
    "MethodTrainer",
    "apply_rdrop",
    "apply_r3f",
]
