#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Attention Configure Operation Contract and Implementation.

Specification Section 6.2: attention.configure@1.0.0
Configures runtime attention hyperparameters such as block size / chunk size,
clamping bounds, numerical stability parameters, and dynamic auto-tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.errors import OperationError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Attention.Configure")

CONTRACT = OperationContract(
    name="attention.configure",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "properties": {
            "chunk_size": {"type": "integer", "minimum": 16},
            "clamp_min": {"type": "number"},
            "clamp_max": {"type": "number"},
            "auto_tune": {"type": "boolean"},
            "auto_tune_seq_len": {"type": "integer", "minimum": 16},
        },
        "additionalProperties": False,
    },
    preconditions=["model.has_attention"],
    postconditions=["attention_parameters_configured"],
    declares_mutations=["*.attn.config"],
    deterministic_given_seed=True,
    composable_with_self=False,
)

CANDIDATE_BLOCK_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256)


def auto_tune_block_size(
    module: nn.Module,
    seq_len: int = 512,
    candidates: Tuple[int, ...] = CANDIDATE_BLOCK_SIZES,
) -> int:
    """Auto-tunes dynamic block / chunk size based on hardware throughput or sequence constraints.

    Evaluates valid chunk candidates given sequence length and hardware architecture.
    Prefers cache-aligned power-of-two blocks matching L1/L2 and warp bounds.

    Args:
        module: Attention module or parent layer to tune.
        seq_len: Target reference sequence length for auto-tuning.
        candidates: Candidate block sizes to evaluate.

    Returns:
        Optimal integer block size.
    """
    valid_candidates: List[int] = [c for c in candidates if c <= max(16, seq_len)]
    if not valid_candidates:
        return 64

    # Select the optimal block size that maximizes throughput while avoiding memory blowup
    # Hardware heuristic: on CUDA prefer 64 or 128 depending on seq_len; on CPU prefer 32 or 64.
    device_type = "cpu"
    try:
        first_param = next(module.parameters(), None)
        if first_param is not None:
            device_type = first_param.device.type
    except Exception:
        device_type = "cpu"

    if device_type == "cuda":
        preferred = [128, 64, 256, 32, 16]
    else:
        preferred = [64, 32, 128, 16, 256]

    for pref in preferred:
        if pref in valid_candidates:
            return pref

    return valid_candidates[-1]


def _find_attention_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Recursively locates attention modules within model.

    Args:
        model: Root PyTorch module.

    Returns:
        List of (path, module) tuples.
    """
    attn_modules: List[Tuple[str, nn.Module]] = []
    attn_keywords = ("self_attn", "attn", "attention", "cross_attn", "encdec_attn")

    for name, module in model.named_modules():
        if module is model:
            continue
        lower_name = name.lower()
        module_cls_name = module.__class__.__name__.lower()
        if any(kw in lower_name for kw in attn_keywords) or "attention" in module_cls_name:
            # Check if this module is a leaf attention module or container
            has_sub_attn = any(
                any(kw in sub_name.lower() for kw in attn_keywords)
                for sub_name, _ in module.named_children()
            )
            if not has_sub_attn or hasattr(module, "chunk_size") or hasattr(module, "block_size"):
                attn_modules.append((name, module))

    return attn_modules


@register_operation(override=True)
class AttentionConfigureOperation(Operation):
    """Attention hyperparameter configuration and block size auto-tuning operation.

    Configures runtime parameters across all attention submodules in the network:
    - Dynamic chunk size and block size selection
    - Numerical activation clamping bounds (clamp_min, clamp_max)
    - Auto-tuning based on target sequence length and hardware architecture.
    """

    def __init__(self) -> None:
        """Initializes the AttentionConfigureOperation with canonical contract."""
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Applies hyperparameter configuration to model attention blocks.

        Args:
            model: PyTorch model containing attention modules.
            config: Configuration dictionary matching params_schema.

        Returns:
            Tuple of (configured_model, execution_evidence_dict).

        Raises:
            OperationError: If model has no identifiable attention modules.
        """
        attn_modules = _find_attention_modules(model)
        if not attn_modules:
            # Check if root model itself has attention attributes
            if hasattr(model, "chunk_size") or hasattr(model, "block_size") or hasattr(model, "q_proj"):
                attn_modules = [("root", model)]
            else:
                raise OperationError("Precondition failed: model.has_attention — no attention blocks found.")

        chunk_size: Optional[int] = config.get("chunk_size")
        clamp_min: Optional[float] = config.get("clamp_min")
        clamp_max: Optional[float] = config.get("clamp_max")
        auto_tune: bool = config.get("auto_tune", False)
        seq_len: int = config.get("auto_tune_seq_len", 512)

        configured_paths: List[str] = []
        tuned_sizes: Dict[str, int] = {}

        for path, mod in attn_modules:
            effective_chunk_size = chunk_size

            if auto_tune:
                effective_chunk_size = auto_tune_block_size(mod, seq_len=seq_len)
                tuned_sizes[path] = effective_chunk_size

            if effective_chunk_size is not None:
                if hasattr(mod, "chunk_size"):
                    mod.chunk_size = effective_chunk_size
                if hasattr(mod, "block_size"):
                    mod.block_size = effective_chunk_size
                # If wrapped in UniversalAttentionBridge or has core attention
                if hasattr(mod, "attention") and hasattr(mod.attention, "chunk_size"):
                    mod.attention.chunk_size = effective_chunk_size
                if hasattr(mod, "new_attention") and hasattr(mod.new_attention, "chunk_size"):
                    mod.new_attention.chunk_size = effective_chunk_size

            if clamp_min is not None:
                mod.clamp_min = clamp_min
                if hasattr(mod, "attention"):
                    mod.attention.clamp_min = clamp_min
            if clamp_max is not None:
                mod.clamp_max = clamp_max
                if hasattr(mod, "attention"):
                    mod.attention.clamp_max = clamp_max

            # Record configured state in module attributes
            if not hasattr(mod, "_attention_config"):
                mod._attention_config = {}
            mod._attention_config.update({
                "chunk_size": effective_chunk_size,
                "clamp_min": clamp_min,
                "clamp_max": clamp_max,
                "auto_tuned": auto_tune,
            })
            configured_paths.append(path)

        evidence: Dict[str, Any] = {
            "params_used": config,
            "operation_version": self.version,
            "output_state": self.output_state,
            "configured_modules_count": len(configured_paths),
            "configured_paths": configured_paths,
            "tuned_sizes": tuned_sizes,
        }
        logger.info(
            "AttentionConfigureOperation: Configured %d attention modules (auto_tune=%s).",
            len(configured_paths),
            auto_tune,
        )
        return model, evidence
