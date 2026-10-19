# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Trainer Methods Subpackage.
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

from neuralops.trainer.methods import fisher
from neuralops.trainer.methods import gelora
from neuralops.trainer.methods import llrd
from neuralops.trainer.methods import lora
from neuralops.trainer.methods import mixout
from neuralops.trainer.methods import rdrop_r3f
from neuralops.trainer.methods import registry
