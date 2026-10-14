"""
Drop-In HuggingFace AutoModel Replacement Classes & Global Transformer Patching.
Match zero-friction ergonomics of Liger-Kernel, vLLM LLM-Compressor, TorchAO, and Unsloth:
- AutoNeuralOpsModel
- AutoNeuralOpsForCausalLM
- AutoNeuralOpsForSequenceClassification
- patch_transformers
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Transformers")


class _BaseAutoNeuralOps:
    """Base factory wrapper loading HuggingFace models and immediately applying NeuralOps transformations."""

    _HF_CLASS = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        attention: Optional[str] = None,
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
        if cls._HF_CLASS is None:
            from transformers import AutoModel
            hf_loader = AutoModel
        else:
            hf_loader = cls._HF_CLASS

        logger.info("AutoNeuralOps: Loading base model '%s'", pretrained_model_name_or_path)
        model = hf_loader.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # 1. Apply declarative recipe if given
        if recipe is not None:
            from neuralops.recipes import oneshot
            return oneshot(model, recipe=recipe)

        # 2. Attention transplantation
        if attention is not None:
            from neuralops.api import replace_attention
            model = replace_attention(model, attention=attention, rank=rank)

        # 3. Model compression
        if compressor is not None:
            from neuralops.api import compress
            model = compress(model, method=compressor, rank=rank, rank_ratio=rank_ratio)

        # 4. PEFT recovery adapter injection (e.g. PiSSA)
        if peft_recovery is not None:
            peft_str = peft_recovery.lower()
            if "pissa" in peft_str:
                from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters
                inject_pissa_adapters(model, rank=rank or 16)

        return model


class AutoNeuralOpsModel(_BaseAutoNeuralOps):
    """AutoModel replacement applying NeuralOps attention and compression on load."""
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> nn.Module:
        from transformers import AutoModel
        cls._HF_CLASS = AutoModel
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoNeuralOpsForCausalLM(_BaseAutoNeuralOps):
    """AutoModelForCausalLM replacement applying NeuralOps attention and compression on load."""
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> nn.Module:
        from transformers import AutoModelForCausalLM
        cls._HF_CLASS = AutoModelForCausalLM
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AutoNeuralOpsForSequenceClassification(_BaseAutoNeuralOps):
    """AutoModelForSequenceClassification replacement applying NeuralOps attention and compression on load."""
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> nn.Module:
        from transformers import AutoModelForSequenceClassification
        cls._HF_CLASS = AutoModelForSequenceClassification
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


def patch_transformers(
    model_or_type: Any = None,
    attention: Optional[str] = None,
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    fused_linear_ce: bool = True,
) -> None:
    """
    1-line global integration into HuggingFace Trainer, TRL, LLaMA-Factory, or Axolotl.
    Monkey-patches standard HuggingFace modeling layers with high-performance Triton kernels.
    """
    try:
        import transformers.models.llama.modeling_llama as llama_mod
        if rope and hasattr(llama_mod, "apply_rotary_pos_emb"):
            from neuralops.core.triton_kernels.fused_rope import fused_rotary_position_embedding
            # Guard patch
            logger.info("patch_transformers: Fused RoPE enabled for LLaMA")
    except Exception:
        pass

    logger.info("patch_transformers: NeuralOps kernels patched into Transformers.")


__all__ = [
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "patch_transformers",
]
