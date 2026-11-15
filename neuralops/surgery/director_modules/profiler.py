#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Architecture Profiler and Topology Inspector.

Classifies arbitrary PyTorch models into 9 canonical families and detects
projection layout, positional embedding topology, and invariant properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.surgery.director_modules.embedding import (
    get_input_embedding_module,
    get_output_embedding_module,
)

logger = logging.getLogger("neuralops.surgery.director.profiler")


@dataclass
class ArchitectureProfile:
    """Structured architectural profile classifying model topology and invariant contracts."""

    family: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    has_qk_norm: bool
    softcapping_factor: Optional[float]
    is_encoder_decoder: bool
    has_tied_embeddings: bool
    is_quantized: bool
    num_kv_heads: Optional[int] = None
    is_encoder: bool = False
    is_mla: bool = False
    positional_topology: Optional[str] = None
    positional_params: Optional[Dict[str, Any]] = None
    projection_layout: Optional[str] = None
    extra_attributes: Optional[Dict[str, Any]] = None


def detect_projection_layout(module_or_model: nn.Module) -> str:
    """
    Phase 042: Deep inspection of projection topology.
    Detects whether attention projections are:
      - 'decoupled': separate Q, K, V projections (LLaMA, Mistral, BERT)
      - 'packed': fused QKV projection (Falcon, ViT, nn.MultiheadAttention)
      - 'transposed_conv1d': transposed 1D convolutional projections (GPT-2)
      - 'unknown': when projections cannot be identified
    """
    cls_name = module_or_model.__class__.__name__
    if hasattr(module_or_model, "c_attn") or "conv1d" in cls_name.lower():
        return "transposed_conv1d"
    if (
        hasattr(module_or_model, "qkv")
        or hasattr(module_or_model, "query_key_value")
        or hasattr(module_or_model, "in_proj_weight")
    ):
        return "packed"
    if (
        (hasattr(module_or_model, "q_proj") and hasattr(module_or_model, "k_proj"))
        or (hasattr(module_or_model, "query") and hasattr(module_or_model, "key"))
        or (hasattr(module_or_model, "q_lin") and hasattr(module_or_model, "k_lin"))
        or (hasattr(module_or_model, "q") and hasattr(module_or_model, "k"))
    ):
        return "decoupled"

    named_mods = dict(module_or_model.named_modules())
    if any("c_attn" in n or "Conv1D" in m.__class__.__name__ for n, m in named_mods.items()):
        return "transposed_conv1d"
    if any(any(f in n for f in ("query_key_value", "qkv", "in_proj_weight")) for n in named_mods):
        return "packed"
    if any(any(f in n for f in ("q_proj", "k_proj", "v_proj")) for n in named_mods):
        return "decoupled"
    if any(any(f in n for f in ("query", "key", "value")) for n in named_mods):
        return "decoupled"

    return "unknown"


_detect_projection_layout = detect_projection_layout


def _extract_linear_w_b(
    mod: Union[nn.Module, nn.Parameter, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extracts standard (out_features, in_features) weight and optional bias."""
    if isinstance(mod, nn.Linear):
        return mod.weight, mod.bias
    if isinstance(mod, (nn.Parameter, torch.Tensor)):
        return mod, None
    raw_w = getattr(mod, "weight", None)
    raw_b = getattr(mod, "bias", None)
    if raw_w is not None and isinstance(raw_w, torch.Tensor):
        if hasattr(mod, "nf") or "conv1d" in mod.__class__.__name__.lower():
            if raw_w.ndim == 2 and raw_w.shape[0] != raw_w.shape[1]:
                return raw_w.t().contiguous(), raw_b
        return raw_w, raw_b
    raise ValueError(f"Cannot extract weight and bias from module {mod}")


def extract_qkv_weights(
    module_or_model: nn.Module,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Phase 042: Deep Inspection Engine for Packed vs Decoupled Projections.
    Accurately extracts and slices Q, K, V, and Output projection weights and biases:
      1. Decoupled projections (q_proj, k_proj, v_proj, o_proj)
      2. Packed projections (query_key_value, qkv, in_proj_weight) with MHA, GQA, or MQA slicing
      3. Transposed Conv1D projections (c_attn, c_proj) with shape (in_features, out_features)

    Returns:
      (W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o) with zero parameter leakage or transposition errors.
    """
    # 1. Transposed Conv1D (GPT-2 style)
    if hasattr(module_or_model, "c_attn"):
        c_attn = module_or_model.c_attn
        c_proj = getattr(module_or_model, "c_proj", None)
        raw_w = getattr(c_attn, "weight", None)
        raw_b = getattr(c_attn, "bias", None)
        if raw_w is None:
            raise ValueError(f"Cannot extract weight from {c_attn}")
        if raw_w.ndim == 2 and (raw_w.shape[0] < raw_w.shape[1] or "conv1d" in c_attn.__class__.__name__.lower()):
            W_attn = raw_w.t().contiguous()
        else:
            W_attn = raw_w

        W_o, b_o = None, None
        if c_proj is not None:
            w_o_raw = getattr(c_proj, "weight", None)
            if w_o_raw is not None:
                if w_o_raw.ndim == 2 and "conv1d" in c_proj.__class__.__name__.lower():
                    W_o = w_o_raw.t().contiguous()
                else:
                    W_o = w_o_raw
            b_o = getattr(c_proj, "bias", None)

        total_qkv = W_attn.shape[0]
        split_dim = total_qkv // 3
        W_q = W_attn[0:split_dim, :]
        W_k = W_attn[split_dim : 2 * split_dim, :]
        W_v = W_attn[2 * split_dim : 3 * split_dim, :]
        if raw_b is not None:
            b_q = raw_b[0:split_dim]
            b_k = raw_b[split_dim : 2 * split_dim]
            b_v = raw_b[2 * split_dim : 3 * split_dim]
        else:
            b_q, b_k, b_v = None, None, None
        return W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o

    # 2. Packed Projections (qkv, query_key_value, in_proj_weight)
    packed_attr = None
    for cand in ("query_key_value", "qkv", "in_proj_weight"):
        if hasattr(module_or_model, cand):
            packed_attr = cand
            break

    if packed_attr is not None:
        packed_mod = getattr(module_or_model, packed_attr)
        W_packed, b_packed = _extract_linear_w_b(packed_mod)
        total_out = W_packed.shape[0]

        # Determine output projection if present
        W_o, b_o = None, None
        for cand_o in ("dense", "out_proj", "o_proj", "proj"):
            if hasattr(module_or_model, cand_o):
                mod_o = getattr(module_or_model, cand_o)
                try:
                    W_o, b_o = _extract_linear_w_b(mod_o)
                    break
                except Exception:
                    pass

        # Handle GQA / MQA slicing if num_kv_heads != num_heads
        if num_heads is not None and num_kv_heads is not None and head_dim is not None:
            q_dim = num_heads * head_dim
            kv_dim = num_kv_heads * head_dim
            if q_dim + 2 * kv_dim == total_out:
                W_q = W_packed[:q_dim, :]
                W_k = W_packed[q_dim : q_dim + kv_dim, :]
                W_v = W_packed[q_dim + kv_dim :, :]
                if b_packed is not None:
                    b_q = b_packed[:q_dim]
                    b_k = b_packed[q_dim : q_dim + kv_dim]
                    b_v = b_packed[q_dim + kv_dim :]
                else:
                    b_q, b_k, b_v = None, None, None
                return W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o

        # Equal 3-way split default (MHA)
        split_dim = total_out // 3
        W_q = W_packed[0:split_dim, :]
        W_k = W_packed[split_dim : 2 * split_dim, :]
        W_v = W_packed[2 * split_dim : 3 * split_dim, :]
        if b_packed is not None:
            b_q = b_packed[0:split_dim]
            b_k = b_packed[split_dim : 2 * split_dim]
            b_v = b_packed[2 * split_dim : 3 * split_dim]
        else:
            b_q, b_k, b_v = None, None, None
        return W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o

    # 3. Decoupled Projections (q_proj, k_proj, v_proj, o_proj)
    q_mod, k_mod, v_mod, o_mod = None, None, None, None
    if hasattr(module_or_model, "q_proj") and hasattr(module_or_model, "k_proj") and hasattr(module_or_model, "v_proj"):
        q_mod = module_or_model.q_proj
        k_mod = module_or_model.k_proj
        v_mod = module_or_model.v_proj
        o_mod = getattr(module_or_model, "o_proj", getattr(module_or_model, "out_proj", None))
    elif hasattr(module_or_model, "query") and hasattr(module_or_model, "key") and hasattr(module_or_model, "value"):
        q_mod = module_or_model.query
        k_mod = module_or_model.key
        v_mod = module_or_model.value
        o_mod = getattr(module_or_model, "out_proj", getattr(module_or_model, "dense", None))
    elif hasattr(module_or_model, "q_lin") and hasattr(module_or_model, "k_lin") and hasattr(module_or_model, "v_lin"):
        q_mod = module_or_model.q_lin
        k_mod = module_or_model.k_lin
        v_mod = module_or_model.v_lin
        o_mod = getattr(module_or_model, "out_lin", None)
    elif hasattr(module_or_model, "q") and hasattr(module_or_model, "k") and hasattr(module_or_model, "v"):
        q_mod = module_or_model.q
        k_mod = module_or_model.k
        v_mod = module_or_model.v
        o_mod = getattr(module_or_model, "o", None)

    if q_mod is not None and k_mod is not None and v_mod is not None:
        W_q, b_q = _extract_linear_w_b(q_mod)
        W_k, b_k = _extract_linear_w_b(k_mod)
        W_v, b_v = _extract_linear_w_b(v_mod)
        W_o, b_o = _extract_linear_w_b(o_mod) if o_mod is not None else (None, None)
        return W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o

    # 4. Search submodules for target attention blocks
    from neuralops.surgery.scanner import UniversalArchitectureScanner

    targets = UniversalArchitectureScanner.scan(module_or_model)
    if targets and targets[0].extracted_qkv is not None:
        return targets[0].extracted_qkv

    raise ValueError(f"Could not extract QKV projections from module {module_or_model}")


def inspect_positional_topology(model: nn.Module) -> Tuple[str, Dict[str, Any]]:
    """
    Phase 043: Positional Embedding Topology Inspector.
    Detects positional topology across model hierarchy:
      - 'rope': Rotary Position Embedding (with base theta and scaling)
      - 'alibi': Attention with Linear Biases (slopes per head)
      - 'learned_absolute': Learned Absolute Position Embeddings (BERT, GPT-2, ViT)
      - 'relative_position_buckets': T5 relative position bucket offsets
      - 'none': When no positional embeddings are identified
    """
    cfg = getattr(model, "config", None)
    named_modules = dict(model.named_modules())
    model_cls_name = model.__class__.__name__.lower()
    config_type = str(getattr(cfg, "model_type", "")).lower()

    # 1. Check for RoPE (LLaMA, Mistral, Qwen, Gemma, etc.)
    has_rope_attr = (
        hasattr(cfg, "rope_theta")
        or hasattr(cfg, "rope_scaling")
        or any("rotary" in n.lower() for n in named_modules)
        or any("rope" in n.lower() for n in named_modules)
        or any(arch in model_cls_name or arch in config_type for arch in ("llama", "mistral", "qwen", "gemma"))
    )
    if has_rope_attr:
        base_theta = float(getattr(cfg, "rope_theta", 10000.0))
        rope_scaling = getattr(cfg, "rope_scaling", None)
        max_pos = getattr(cfg, "max_position_embeddings", None)
        return "rope", {
            "base_theta": base_theta,
            "rope_scaling": rope_scaling,
            "max_position_embeddings": max_pos,
        }

    # 2. Check for ALiBi (Bloom, MPT, Falcon-7B)
    has_alibi = (
        bool(getattr(cfg, "alibi", False))
        or bool(getattr(cfg, "use_alibi", False))
        or any("alibi" in n.lower() for n in named_modules)
        or "bloom" in model_cls_name
        or "bloom" in config_type
    )
    if has_alibi:
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 8))
        return "alibi", {
            "num_heads": num_heads,
            "has_alibi": True,
        }

    # 3. Check for Relative Position Buckets (T5)
    is_t5 = (
        getattr(cfg, "is_encoder_decoder", False)
        or "t5" in model_cls_name
        or "t5" in config_type
        or any("relative_attention_bias" in n for n in named_modules)
    )
    if is_t5:
        num_buckets = getattr(cfg, "relative_attention_num_buckets", 32)
        max_dist = getattr(cfg, "relative_attention_max_distance", 128)
        return "relative_position_buckets", {
            "num_buckets": num_buckets,
            "max_distance": max_dist,
            "bidirectional": getattr(cfg, "is_decoder", False) is False,
        }

    # 4. Check for Learned Absolute (BERT, RoBERTa, GPT-2, ViT)
    has_abs = (
        getattr(cfg, "position_embedding_type", "") == "absolute"
        or any("wpe" in n or "position_embeddings" in n or "pos_embed" in n for n in named_modules)
        or any(arch in model_cls_name or arch in config_type for arch in ("bert", "roberta", "gpt2", "vit"))
    )
    if has_abs:
        max_pos = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or 512
        emb_dim = (
            getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None) or 768
        )
        return "learned_absolute", {
            "max_position_embeddings": max_pos,
            "embedding_dim": emb_dim,
        }

    return "none", {}


_inspect_positional_topology = inspect_positional_topology


def profile_model_topology(model: nn.Module) -> ArchitectureProfile:
    """
    Classifies model architecture into one of 9 canonical families:
    1. llama: Standard causal decoder with RoPE and RMSNorm (LLaMA, Mistral, Qwen).
    2. gemma2: QK-Norm presence, logit soft-capping factor.
    3. deepseek_mla: Multi-head Latent Attention with compressed KV representations.
    4. bert: Bidirectional encoder with absolute position embeddings.
    5. t5: Encoder-decoder with static cross-attention state.
    6. falcon: Fused QKV or transposed Conv1D projections (Falcon, GPT-2).
    7. vit: Vision Transformer with patch embedding projections.
    8. jamba: Alternating Transformer and Mamba SSM layers.
    9. quantized: Quantized linear base projections (4-bit/8-bit).
    """
    cfg = getattr(model, "config", None)
    num_layers = (
        getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None) or getattr(cfg, "n_layer", 12)
    )
    hidden_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or getattr(cfg, "n_embd", 768)
    num_heads = (
        getattr(cfg, "num_attention_heads", None) or getattr(cfg, "num_heads", None) or getattr(cfg, "n_head", 12)
    )
    num_kv_heads = (
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "num_kv_heads", None)
        or getattr(cfg, "n_head_kv", None)
        or num_heads
    )
    head_dim = getattr(cfg, "head_dim", None) or (hidden_dim // max(num_heads, 1))

    named_modules = dict(model.named_modules())
    model_cls_name = model.__class__.__name__.lower()
    config_type = str(getattr(cfg, "model_type", "")).lower()

    # Detect QK-Norm (Gemma-2 Law 1)
    has_qk_norm = any("q_norm" in name or "k_norm" in name for name in named_modules)

    # Detect Soft-Capping factor (Gemma-2 Law 2)
    softcapping_factor: Optional[float] = None
    if cfg is not None:
        softcapping_factor = getattr(cfg, "attn_logit_softcapping", None) or getattr(
            cfg, "final_logit_softcapping", None
        )
    if softcapping_factor is None:
        for mod in named_modules.values():
            if hasattr(mod, "attn_logit_softcapping"):
                val = mod.attn_logit_softcapping
                if val is not None:
                    softcapping_factor = float(val)
                    break

    # Detect Encoder-Decoder (T5 Law 4)
    is_encoder_decoder = getattr(cfg, "is_encoder_decoder", False)
    if not is_encoder_decoder:
        is_encoder_decoder = (
            (
                hasattr(model, "encoder")
                and hasattr(model, "decoder")
                and ("t5" in model_cls_name or "t5" in config_type or any("encdec" in n.lower() for n in named_modules))
            )
            or "t5" in model_cls_name
            or "t5" in config_type
        )

    # Detect Tied Embeddings (Law 8)
    has_tied = False
    inp_emb = get_input_embedding_module(model)
    out_head = get_output_embedding_module(model)
    if inp_emb is not None and out_head is not None:
        inp_w = getattr(inp_emb, "weight", None)
        out_w = getattr(out_head, "weight", None)
        if inp_w is not None and out_w is not None:
            has_tied = id(inp_w) == id(out_w)

    # Detect Quantized Base (Phase 050)
    is_quantized = any(
        "linear4bit" in mod.__class__.__name__.lower()
        or "linear8bit" in mod.__class__.__name__.lower()
        or "wqlinear" in mod.__class__.__name__.lower()
        or hasattr(mod, "qweight")
        for mod in named_modules.values()
    )

    # Detect Bidirectional Encoder vs Causal Decoder
    is_encoder = (
        "bert" in model_cls_name
        or "roberta" in model_cls_name
        or "bert" in config_type
        or "vit" in model_cls_name
        or "vit" in config_type
    ) and not is_encoder_decoder

    # Detect MLA (Phase 044)
    is_mla = (
        getattr(cfg, "multi_head_latent_attention", False)
        or any(
            "kv_b_proj" in name or "q_b_proj" in name or "kv_a_proj_with_mqa" in name or "mla" in name
            for name in named_modules
        )
        or "deepseek" in model_cls_name
        or "deepseek" in config_type
    )

    # Jamba Layer Identification (Phase 049)
    jamba_attn_indices: List[int] = []
    if any("mamba" in name.lower() or "mamba" in mod.__class__.__name__.lower() for name, mod in named_modules.items()):
        # Scan layer indices containing self_attn
        for idx, (name, mod) in enumerate(named_modules.items()):
            if "self_attn" in name or "attention" in name.lower():
                jamba_attn_indices.append(idx)

    # Detect 9 Canonical Families
    if is_quantized:
        family = "quantized"
    elif (
        jamba_attn_indices
        or "jamba" in model_cls_name
        or "jamba" in config_type
        or any("mamba" in name.lower() for name in named_modules)
    ):
        family = "jamba"
    elif is_mla:
        family = "deepseek_mla"
    elif (
        has_qk_norm
        or (softcapping_factor is not None and softcapping_factor > 0)
        or "gemma" in model_cls_name
        or "gemma" in config_type
    ):
        family = "gemma2"
    elif is_encoder_decoder:
        family = "t5"
    elif any("patch_embed" in name for name in named_modules) or "vit" in model_cls_name or "vit" in config_type:
        family = "vit"
    elif (
        any("query_key_value" in name or "c_attn" in name for name in named_modules)
        or "falcon" in model_cls_name
        or "gpt2" in model_cls_name
    ):
        family = "gpt2" if ("gpt2" in model_cls_name or "gpt2" in config_type) else "falcon"
    elif is_encoder:
        family = "bert"
    else:
        family = "llama"

    pos_topology, pos_params = inspect_positional_topology(model)
    proj_layout = detect_projection_layout(model)

    profile = ArchitectureProfile(
        family=family,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        has_qk_norm=has_qk_norm,
        softcapping_factor=softcapping_factor,
        is_encoder_decoder=is_encoder_decoder,
        has_tied_embeddings=has_tied,
        is_quantized=is_quantized,
        num_kv_heads=num_kv_heads,
        is_encoder=is_encoder,
        is_mla=is_mla,
        positional_topology=pos_topology,
        positional_params=pos_params,
        projection_layout=proj_layout,
        extra_attributes={
            "model_class": model.__class__.__name__,
            "config_type": config_type,
            "selective_attention_layer_indices": jamba_attn_indices,
            "num_kv_heads": num_kv_heads,
            "positional_topology": pos_topology,
            "positional_params": pos_params,
            "projection_layout": proj_layout,
        },
    )

    logger.info(
        "Profiled model topology: family=%s, layers=%d, hidden_dim=%d, heads=%d, head_dim=%d, QKNorm=%s, Tied=%s, Quant=%s",
        profile.family,
        profile.num_layers,
        profile.hidden_dim,
        profile.num_heads,
        profile.head_dim,
        profile.has_qk_norm,
        profile.has_tied_embeddings,
        profile.is_quantized,
    )
    return profile
