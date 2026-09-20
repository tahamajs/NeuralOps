"""Mamba Injection Module.

Provides injection hooks to wire Bidirectional Mamba into transformer layers
with automated weight copying from pretrained self-attention projections.
"""

import logging
from typing import Any, Optional, Union

import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("mamba_module_inject")
@ATTENTION_REGISTRY.register("mamba_inject")
@ATTENTION_REGISTRY.register("inject_mamba")
def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[nn.Module, Any]]:
    r"""Safe injection hook to wire Mamba into transformer layers with weight copying."""
    from framework.attentions.mamba import Mamba

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal support
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_mamba(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn

    rank = kwargs.pop("rank", 32)
    d_model = 128
    if cfg is not None:
        d_model = getattr(cfg, "hidden_size", getattr(cfg, "dim", getattr(cfg, "d_model", 128)))
    elif src_attn is not None:
        d_model = (
            getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or getattr(src_attn, "dim", None)
            or (src_attn.q_proj.in_features if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None)
            or (src_attn.q_lin.in_features if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None)
            or 128
        )
    elif target is not None and hasattr(target, "config"):
        d_model = getattr(target.config, "hidden_size", getattr(target.config, "dim", 128))
    elif target is not None and hasattr(target, "hidden_size"):
        d_model = getattr(target, "hidden_size", 128)

    try:
        new_module = Mamba(d_model=d_model, config=cfg, comp_config=comp_config, rank=rank, **kwargs)
    except Exception as e:
        logger.error("Failed to instantiate Mamba: %s", e)
        return None

    if src_attn is not None and hasattr(new_module, "copy_weights"):
        try:
            new_module.copy_weights(src_attn)
        except Exception as e:
            logger.error("copy_weights failed: %s", e)
            return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_module
            else:
                target.attention = new_module
        elif hasattr(target, "self_attn"):
            target.self_attn = new_module
        elif hasattr(target, "self"):
            target.self = new_module
        elif hasattr(target, "attn"):
            target.attn = new_module
        else:
            target.attention = new_module

    return new_module
