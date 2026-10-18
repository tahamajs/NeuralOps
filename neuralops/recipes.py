"""
NeuralOps Transformation Recipes

Declarative one-shot recipe engine and hardware-aware pipeline advisor for model transformation.

Author: Taha Majlesi
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Recipes")

BUILTIN_RECIPES: Dict[str, Dict[str, Any]] = {
    "fast_linear_recovery": {
        "description": "Fast linear attention recovery with GLA and AdaSVD + PiSSA",
        "attention": "gla_attention",
        "compressor": "adasvd",
        "rank_ratio": 0.5,
        "peft_recovery": "pissa",
    },
    "cosformer_encoder_svd": {
        "description": "Cosformer linear attention with CUR compressor",
        "attention": "cosformer",
        "compressor": "cur_compressor",
        "rank_ratio": 0.6,
    },
    "low_vram_6gb": {
        "description": "Low VRAM profile with Cosformer and Rotational SliceGPT INT4",
        "attention": "cosformer",
        "compressor": "rotational_slicegpt_int4",
        "rank_ratio": 0.4,
    },
    "hybrid_jamba_ratio": {
        "description": "Hybrid Jamba-style ratio with ModeGPT",
        "attention": "tri_branch_hybrid",
        "compressor": "modegpt",
        "rank_ratio": 0.5,
    },
    "lossless_svd_80": {
        "description": "High-fidelity Swift-SVD at 80% rank retention with PiSSA",
        "compressor": "swift_svd_act",
        "rank_ratio": 0.8,
        "peft_recovery": "pissa",
    },
    "flagship_2026": {
        "description": "Flagship 2026 recipe with GDN-2 attention and AA-SVD compressor",
        "attention": "gdn2_attention",
        "compressor": "aa_svd_compressor",
        "rank_ratio": 0.5,
    },
}


def oneshot(
    model: nn.Module,
    recipe: Union[str, Dict[str, Any]],
    calibration_data: Any = None,
    **kwargs: Any,
) -> nn.Module:
    """Executes a complete chained transformation recipe on a model in-place."""
    if isinstance(recipe, str):
        recipe_name = recipe.lower()
        if recipe_name not in BUILTIN_RECIPES:
            raise ValueError(f"Unknown builtin recipe '{recipe}'. Available: {list(BUILTIN_RECIPES.keys())}")
        recipe_dict = BUILTIN_RECIPES[recipe_name]
    else:
        recipe_dict = recipe

    logger.info("neuralops.oneshot: Executing recipe %s", recipe_dict)

    attn = recipe_dict.get("attention")
    if attn:
        from neuralops.api import replace_attention
        model = replace_attention(model, attention=attn, rank=recipe_dict.get("rank"))

    comp = recipe_dict.get("compressor")
    if comp:
        from neuralops.api import compress
        model = compress(
            model,
            method=comp,
            rank=recipe_dict.get("rank"),
            rank_ratio=recipe_dict.get("rank_ratio", 0.5),
        )

    peft = recipe_dict.get("peft_recovery")
    if peft and "pissa" in str(peft).lower():
        from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters
        inject_pissa_adapters(model, rank=recipe_dict.get("rank", 16))

    return model


def recommend(
    model_or_config: Any = None,
    target_vram_gb: Optional[float] = None,
    task: str = "causal_lm",
    seq_len: int = 4096,
    gpu_memory_gb: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Inspects the active GPU and returns tailored architecture, compressor, and recipe advice."""
    if gpu_memory_gb is not None:
        target_vram_gb = gpu_memory_gb
    sm_major = 7
    if torch.cuda.is_available():
        sm_major = torch.cuda.get_device_capability()[0]

    if sm_major >= 8:  # Ada Lovelace, Hopper, Ampere
        rec = {
            "recommended_attention": "gdn2_attention" if seq_len >= 2048 else "gla_attention",
            "recommended_compressor": "swift_svd_act",
            "recommended_recipe": "flagship_2026",
            "optimal_rank": 64,
            "optimal_chunk_size": 64,
            "hardware_tier": f"SM {sm_major}.x (Modern Tensor Core / Full Triton)",
        }
    else:  # Pascal / Legacy
        rec = {
            "recommended_attention": "cosformer",
            "recommended_compressor": "rotational_slicegpt_int4",
            "recommended_recipe": "low_vram_6gb",
            "optimal_rank": 32,
            "optimal_chunk_size": 32,
            "hardware_tier": f"SM {sm_major}.x (Legacy Pascal / Reference Fallback)",
        }

    return rec


__all__ = [
    "BUILTIN_RECIPES",
    "oneshot",
    "recommend",
]
