"""Ecosystem and framework integrations for NeuralOps."""
from __future__ import annotations

from neuralops.integrations.hf_automodel import register_hf_automodel, AutoNeuralOpsModel
from neuralops.integrations.distributed_compile import (
    compile_attention_module,
    get_deepspeed_config,
    apply_fsdp_wrapping,
)

__all__ = [
    "register_hf_automodel",
    "AutoNeuralOpsModel",
    "compile_attention_module",
    "get_deepspeed_config",
    "apply_fsdp_wrapping",
]

# Auto-wired package members
from neuralops.integrations import distributed_compile
from neuralops.integrations import hf_automodel
from neuralops.integrations import serving
