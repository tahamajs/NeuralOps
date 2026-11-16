# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention surgery operations: replace_attention, transplant_attention, compose_attention.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger("NeuralOps.API")


def replace_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    validation_level: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Drop-in replace attention layers in any PyTorch / HuggingFace model.

    Records the operation in `model._neuralops_metadata`.

    Args:
        model: HuggingFace or PyTorch Transformer model (GPT-2, LLaMA, Mistral, BERT, etc.).
        attention: Attention mechanism name from `neuralops.list_attentions()`
                   (e.g., "cosformer", "mamba_block", "gqa_attention", "svd_linear_attention").
        rank: Rank parameter for low-rank or SVD-based attention variants (default: None).
        copy_weights: Whether to transplant weights from the original model (default: True).
        layers: Optional list of layer indices to replace (default: None replaces all layers).
        attention_type: Alias for attention.
        validation_level: Optional validation level ('strict', 'lenient', or None) for checking surgery integrity.
        **kwargs: Additional parameters forwarded to the attention module constructor.

    Returns:
        The model with attention layers replaced in-place.
    """
    if model is None:
        raise ValueError("model cannot be None.")

    target_mod = kwargs.pop("target_module", None) or kwargs.pop("new_attention_cls", None)
    if isinstance(target_mod, str) and not attention:
        attention = target_mod
        target_mod = None
    elif isinstance(attention, nn.Module):
        target_mod = attention
        attention = None

    logger.info(
        "Replacing attention with '%s' (copy_weights=%s)",
        attention or (type(target_mod).__name__ if target_mod else "custom"),
        copy_weights,
    )

    layers = layers or kwargs.pop("layer_indices", None)
    if layers == "all":
        layers = None

    if target_mod is not None:
        if model is None:
            return None
        from neuralops.core.architecture import get_attention_modules
        from neuralops.core.model_utils import replace_module

        modules_list = list(get_attention_modules(model))
        for layer_i, (name, old_attn) in enumerate(modules_list):
            if layers is not None and layer_i not in layers:
                continue
            import copy

            mod_copy = copy.deepcopy(target_mod)
            old_param = next(old_attn.parameters(), None)
            if old_param is not None:
                mod_copy = mod_copy.to(device=old_param.device, dtype=old_param.dtype)
            from neuralops.utils.replace_attention import UniversalAttentionWrapper

            wrapped = UniversalAttentionWrapper(mod_copy, layer_name=name, return_tuple=True)
            replace_module(model, name, wrapped)

        meta_entry: Dict[str, Any] = {
            "op": "replace",
            "type": type(target_mod).__name__,
            "kwargs": {
                "rank": rank,
                "copy_weights": copy_weights,
                "layers": layers,
                **kwargs,
            },
        }
        if not hasattr(model, "_neuralops_metadata") or model._neuralops_metadata is None:
            model._neuralops_metadata = []
        model._neuralops_metadata.append(meta_entry)
        from neuralops.utils.hub_integration import attach_attention_methods

        return attach_attention_methods(model)

    attn_name = (
        attention
        or attention_type
        or kwargs.pop("attention_key", None)
        or kwargs.pop("target_attention", None)
        or kwargs.pop("target", None)
    )
    if not attn_name:
        raise ValueError("Must specify attention or attention_type.")
    if attn_name in ("chunk_gla", "gla"):
        attn_name = "gla_attention"
    if attn_name.startswith("fla:"):
        from neuralops.core.exceptions import IntegrationNotImplementedError
        from neuralops.integrations.fla import SUPPORTED_FLA_ATTENTIONS, fla_is_available

        if not fla_is_available():
            raise IntegrationNotImplementedError(
                f"Cannot replace attention with '{attn_name}': the 'fla' package is not installed. "
                "Install it via: pip install 'neuralops[fla]' (requires Linux + CUDA)."
            )
        if attn_name not in SUPPORTED_FLA_ATTENTIONS:
            raise IntegrationNotImplementedError(
                f"Unsupported FLA attention '{attn_name}'. Supported mechanisms: {SUPPORTED_FLA_ATTENTIONS}"
            )
    elif attn_name not in ATTENTION_REGISTRY:
        from neuralops.core.exceptions import AttentionNotRegisteredError

        raise AttentionNotRegisteredError(
            f"Unknown attention module '{attn_name}'. "
            f"Use neuralops.list_attentions() to see all {len(ATTENTION_REGISTRY)} available modules."
        )

    call_kwargs = dict(kwargs)
    if layers is not None:
        call_kwargs["layer_indices"] = layers
    if validation_level is not None:
        call_kwargs["validation_level"] = validation_level

    hybrid_spec = kwargs.get("hybrid_spec", None)
    if hybrid_spec is not None:
        from neuralops.models.hybrid_spec import normalize_hybrid_spec

        norm_spec = normalize_hybrid_spec(hybrid_spec)
        call_kwargs["hybrid_spec"] = norm_spec

    from neuralops.core._attention_injectors import _inject_attention_loop as inject_attention

    logger.info("Replacing attention with '%s' (rank=%s, copy_weights=%s)", attn_name, rank, copy_weights)
    out_model = inject_attention(
        model=model,
        attention_type=attn_name,
        rank=rank,
        copy_weights=copy_weights,
        **call_kwargs,
    )

    meta_entry: Dict[str, Any] = {
        "op": "replace",
        "type": attn_name,
        "kwargs": {
            "rank": rank,
            "copy_weights": copy_weights,
            "layers": layers,
            **kwargs,
        },
    }
    try:
        if not hasattr(out_model, "_neuralops_metadata") or out_model._neuralops_metadata is None:
            out_model._neuralops_metadata = []
        out_model._neuralops_metadata.append(meta_entry)
    except Exception as e:
        logger.error("Failed to register _neuralops_metadata on model %s: %s", type(out_model).__name__, e)
        raise

    from neuralops.export.surgery_manifest import record_attention_surgery

    record_attention_surgery(
        out_model,
        layer_path="all_attention",
        attention_type=attn_name,
        kwargs=meta_entry["kwargs"],
    )

    if hasattr(out_model, "config") and hasattr(out_model.config, "to_dict"):
        if not hasattr(out_model.config, "neuralops_history") or out_model.config.neuralops_history is None:
            out_model.config.neuralops_history = []
        out_model.config.neuralops_history.append(meta_entry)
        try:
            out_model.config.neuralops_attention = attn_name
        except Exception as e:
            logger.error("Failed to set neuralops_attention on model config for %s: %s", type(out_model).__name__, e)
            raise
    elif hasattr(out_model, "config") and out_model.config is not None:
        try:
            out_model.config.neuralops_attention = attn_name
        except Exception as e:
            logger.error("Failed to set neuralops_attention on model config for %s: %s", type(out_model).__name__, e)
            raise

    from neuralops.utils.hub_integration import attach_attention_methods

    return attach_attention_methods(out_model)


def transplant_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    target_attention: Optional[str] = None,
    validation_level: Optional[str] = None,
    calibrate: bool = False,
    calibration_inputs: Optional[Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """Drop-in transplant attention layers with optional closed-form subspace calibration.

    Composes replace_attention with calibrate_attention_subspace when calibrate=True
    or calibration_inputs is provided.
    """
    import copy

    attn = attention or attention_type or target_attention
    should_calibrate = calibrate or (calibration_inputs is not None)
    teacher = None
    if should_calibrate:
        teacher = kwargs.pop("teacher_model", None)
        if teacher is None and model is not None:
            try:
                teacher = copy.deepcopy(model)
            except Exception as e:
                logger.warning("Could not deepcopy model for calibration: %s", e)

    mutated = replace_attention(
        model=model,
        attention=attn,
        rank=rank,
        copy_weights=copy_weights,
        layers=layers,
        validation_level=validation_level,
        **kwargs,
    )

    if should_calibrate and teacher is not None:
        sample = (
            calibration_inputs
            if calibration_inputs is not None
            else kwargs.pop("sample_inputs", kwargs.pop("sample_input", None))
        )
        if sample is not None:
            from neuralops.models.surgery import calibrate_attention_subspace

            ridge_lambda = kwargs.pop("ridge_lambda", 1e-3)
            blend_alpha = kwargs.pop("blend_alpha", 0.25)
            attach_convex_anchor = kwargs.pop("attach_convex_anchor", False)
            mutated = calibrate_attention_subspace(
                teacher_model=teacher,
                student_model=mutated,
                calibration_inputs=sample,
                ridge_lambda=ridge_lambda,
                blend_alpha=blend_alpha,
                attach_convex_anchor=attach_convex_anchor,
            )

    return mutated


# High-Level Public Ergonomics Aliases
transplant = transplant_attention


def compose_attention(
    projection: str = "dense",
    positional: str = "none",
    kernel_map: Union[str, Any] = "softmax",
    mask: str = "causal",
    engine: str = "sdpa",
    embed_dim: int = 768,
    num_heads: int = 12,
    **kwargs: Any,
) -> Any:
    """Compose a modular declarative attention mechanism from customizable primitives.

    Args:
        projection: Linear projection type (e.g. 'dense', 'linear').
        positional: Positional encoding ('none', 'rope', 'alibi').
        kernel_map: Feature kernel mapping ('softmax', 'relu', or custom callable).
        mask: Attention masking ('causal', 'bidirectional', 'none').
        engine: Computation engine ('sdpa', 'exact', 'triton').
        embed_dim: Model hidden representation dimension.
        num_heads: Number of attention query heads.
        **kwargs: Additional parameters forwarded to ComposableAttention.

    Returns:
        Configured ComposableAttention module instance.
    """
    from neuralops.core.composable import compose_attention as _compose

    return _compose(
        projection=projection,
        positional=positional,
        kernel_map=kernel_map,
        mask=mask,
        engine=engine,
        embed_dim=embed_dim,
        num_heads=num_heads,
        **kwargs,
    )
