#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HuggingFace AutoModel Compatibility

Provides compatibility wrappers, attention layer patching, and model loaders for HuggingFace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

__all__ = [
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "register_hf_automodel",
    "patch_model_attention",
    "patch_transformers",
]


def patch_model_attention(model: nn.Module, attention_type: str = "full_attention", **kwargs: Any) -> nn.Module:
    """Replace detected attention modules through the architecture-aware surgery API."""
    from neuralops.api import replace_attention

    return replace_attention(model, attention=attention_type, **kwargs)


class _BaseAutoNeuralOps:
    """Base factory wrapper loading HuggingFace models and immediately applying NeuralOps transformations."""

    _HF_CLASS = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        attention: Optional[str] = None,
        attention_type: Optional[str] = None,
        compressor: Optional[str] = None,
        rank_ratio: float = 0.5,
        rank: Optional[int] = None,
        recipe: Optional[Union[str, Dict[str, Any]]] = None,
        use_fused_kernels: bool = True,
        shadow_twin: bool = False,
        peft_recovery: Optional[str] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Loads model via HuggingFace from_pretrained, then executes specified surgery recipe."""
        attn_to_use = attention or attention_type
        if cls._HF_CLASS is None:
            try:
                from transformers import AutoModel
                hf_loader = AutoModel
            except ImportError:
                hf_loader = None
        else:
            hf_loader = cls._HF_CLASS

        if hf_loader is None:
            raise ImportError(
                "Transformers is required to load a pretrained model. Install it with "
                "`pip install transformers`; NeuralOps will not substitute random weights."
            )
        model = hf_loader.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if recipe is not None:
            from neuralops.recipes import oneshot
            model = oneshot(model, recipe=recipe)

        if attn_to_use is not None:
            from neuralops.api import replace_attention
            model = replace_attention(model, attention=attn_to_use, rank=rank)

        if compressor is not None:
            from neuralops.api import compress
            model = compress(model, method=compressor, rank=rank, rank_ratio=rank_ratio)

        if peft_recovery is not None:
            peft_str = peft_recovery.lower()
            if "pissa" in peft_str:
                from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters
                inject_pissa_adapters(model, rank=rank or 16)

        return model


class AutoNeuralOpsModel(_BaseAutoNeuralOps):
    """AutoModel replacement applying NeuralOps attention and compression on load."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        attention_type: str = "full_attention",
        attention: Optional[str] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> nn.Module:
        try:
            from transformers import AutoModel
            cls._HF_CLASS = AutoModel
        except ImportError:
            cls._HF_CLASS = None
        attn = attention or attention_type
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            attention=attn,
            rank=rank,
            **kwargs,
        )


class AutoNeuralOpsForCausalLM(_BaseAutoNeuralOps):
    """AutoModelForCausalLM replacement applying NeuralOps attention and compression on load."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        attention_type: Optional[str] = None,
        attention: Optional[str] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> nn.Module:
        try:
            from transformers import AutoModelForCausalLM
            cls._HF_CLASS = AutoModelForCausalLM
        except ImportError:
            cls._HF_CLASS = None
        attn = attention or attention_type
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            attention=attn,
            rank=rank,
            **kwargs,
        )


class AutoNeuralOpsForSequenceClassification(_BaseAutoNeuralOps):
    """AutoModelForSequenceClassification replacement applying NeuralOps attention and compression on load."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        attention_type: Optional[str] = None,
        attention: Optional[str] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> nn.Module:
        try:
            from transformers import AutoModelForSequenceClassification
            cls._HF_CLASS = AutoModelForSequenceClassification
        except ImportError:
            cls._HF_CLASS = None
        attn = attention or attention_type
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            attention=attn,
            rank=rank,
            **kwargs,
        )


def register_hf_automodel() -> bool:
    """Hooks NeuralOps configurations into HuggingFace AutoConfig / AutoModel registries if present."""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        from neuralops.models.configuration import NeuralOpsConfig
        from neuralops.models.modeling import NeuralOpsModel, NeuralOpsForCausalLM
        AutoConfig.register("neuralops", NeuralOpsConfig, exist_ok=True)
        AutoModel.register(NeuralOpsConfig, NeuralOpsModel, exist_ok=True)
        AutoModelForCausalLM.register(NeuralOpsConfig, NeuralOpsForCausalLM, exist_ok=True)
        return True
    except ImportError:
        return False


def patch_transformers(
    model_or_type: Any = None,
    attention: Optional[str] = None,
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    fused_linear_ce: bool = True,
) -> None:
    """Reserved compatibility entry point; global Transformers patching is unsupported."""
    raise NotImplementedError(
        "Global Transformers monkey-patching is not implemented. Apply model surgery "
        "explicitly with replace_attention(model, attention=...) instead."
    )
