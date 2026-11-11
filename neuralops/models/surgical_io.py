#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Surgical Model IO and Checkpointing

Model state serialization and weight conversion routines for seamless HuggingFace checkpoint interoperability.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_KNOWN_COMPRESSED_LINEARS = {
    "SVDLinear",
    "FactorizedLinear",
    "DobiSVDLinearLayer",
    "DynamicSVDLinear",
    "ResSVDLinear",
    "SubspaceLinear",
    "A3FactorizedLinear",
    "COMPOTLinear",
    "CURLinear",
    "SVDLLMLinear",
    "CacheNotesLoRALinear",
    "RotationalSliceGPTINT4Linear",
    "KMeansLinear",
    "FAQLinear",
    "BitPackedW4A16Linear",
    "BitPackedLowRankSVDLinear",
    "SliceGPTPCALoRALinear",
    "ADMMMaskedLinear",
    "ASTPrunedLoRALinear",
    "ProxSparsePrunedLoRALinear",
    "MambaShedderPrunedLinear",
    "ARMORLinear",
    "BonsaiPrunedLoRALinear",
}

_WRAPPER_CLASS_NAMES = {
    "AttentionWrapper",
    "ComposableAttention",
    "ReplaceAttention",
    "UniversalAttentionWrapper",
    "QwenAttentionAdapter",
    "BertAttentionAdapter",
}

_ADAPTER_ONLY_WRAPPERS = {
    "AttentionWrapper",
    "ReplaceAttention",
    "UniversalAttentionWrapper",
    "QwenAttentionAdapter",
    "BertAttentionAdapter",
}

_HEAD_LEAF_NAMES = {"classifier", "lm_head", "score", "pooler"}


def _is_vision_tower_path(path: str) -> bool:
    return path == "vision_tower" or path.startswith("vision_tower.") or ".vision_tower." in path


def _matches_any_prefix(key: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        p = prefix.rstrip(".")
        if not p:
            continue
        if key == p or key.startswith(p + "."):
            return True
    return False


def _is_compressed_module(sub: nn.Module) -> bool:
    cls_name = sub.__class__.__name__
    mod_name = sub.__class__.__module__ or ""
    if cls_name in _KNOWN_COMPRESSED_LINEARS:
        return True
    if getattr(sub, "is_compressed", False) or getattr(sub, "is_custom_linear", False):
        return True
    if mod_name.startswith("neuralops.compression"):
        # Avoid matching top-level compressor orchestrator objects if ever attached as nn.Module
        has_tensor_state = any(True for _ in sub.parameters(recurse=False)) or any(
            True for _ in sub.buffers(recurse=False)
        )
        if has_tensor_state:
            return True
    if hasattr(sub, "U") and hasattr(sub, "V"):
        return True
    if hasattr(sub, "weight_u") and hasattr(sub, "weight_v"):
        return True
    if hasattr(sub, "w_down") and hasattr(sub, "w_up"):
        return True
    if hasattr(sub, "qweight") and getattr(sub, "qweight", None) is not None:
        return True
    if hasattr(sub, "pruning_mask") or hasattr(sub, "weight_mask"):
        return True
    if hasattr(sub, "mask") and isinstance(getattr(sub, "mask", None), torch.Tensor):
        return True
    if hasattr(sub, "int_weight") and hasattr(sub, "scale"):
        return True
    if hasattr(sub, "U_packed") or hasattr(sub, "U_q"):
        return True
    if hasattr(sub, "S_sparse") and getattr(sub, "S_sparse", None) is not None:
        return True
    if hasattr(sub, "kept_idx") or hasattr(sub, "keep_idx"):
        return True
    return False


def _is_lora_module(sub: nn.Module) -> bool:
    cls_name = sub.__class__.__name__
    if cls_name == "SimpleLoRALinear" or "lora" in cls_name.lower():
        return True
    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B"):
        if getattr(sub, "lora_A", None) is not None or getattr(sub, "lora_B", None) is not None:
            return True
    return False


def _unwrap_attention_module(sub: nn.Module) -> Tuple[nn.Module, bool]:
    """Return (effective_attention, is_wrapper) for adapters and wrappers."""
    cls_name = sub.__class__.__name__
    wrapped = getattr(sub, "_wrapped_attention", None)
    if isinstance(wrapped, nn.Module):
        return wrapped, True
    if cls_name in ("QwenAttentionAdapter", "BertAttentionAdapter"):
        inner = getattr(sub, "core", None) or getattr(sub, "attn", None)
        if isinstance(inner, nn.Module):
            return inner, True
    if cls_name in _ADAPTER_ONLY_WRAPPERS:
        inner = getattr(sub, "attn", None) or getattr(sub, "core", None)
        if isinstance(inner, nn.Module):
            return inner, True
    attn_attr = getattr(sub, "attn", None)
    if isinstance(attn_attr, nn.Module) and cls_name in _WRAPPER_CLASS_NAMES:
        return attn_attr, True
    return sub, False


def extract_surgical_manifest(model: nn.Module) -> Dict[str, Any]:
    """Walk model tree and extract exact metadata for all surgical modules.

    Captures:
      - Category: 'attention', 'compressed_linear', 'lora_adapter', 'trained_head', 'trained_norm'
      - Exact hierarchical module_path
      - Module class, registration key, constructor parameters, and parameter shapes
    """
    manifest: Dict[str, Any] = {
        "neuralops_version": "2.0.6",
        "model_class": model.__class__.__name__,
        "submodules": {},
        "non_attention_sites": {
            "compressed_linear": [],
            "lora_adapter": [],
            "trained_head": [],
            "trained_norm": [],
        },
    }

    from neuralops.core.base import BaseAttention
    from neuralops.core.registry import ATTENTION_REGISTRY

    registered_attn_classes = tuple(
        c for c in ATTENTION_REGISTRY.values() if isinstance(c, type)
    )
    cataloged_attn_names: list[str] = []
    has_frozen_backbone = any(not p.requires_grad for p in model.parameters())

    for name, sub in model.named_modules():
        if not name:
            continue
        cls_name = sub.__class__.__name__

        if any(name.startswith(p + ".") for p in cataloged_attn_names):
            continue

        is_compressed = _is_compressed_module(sub)
        is_lora = _is_lora_module(sub)
        is_from_compression = (sub.__class__.__module__ or "").startswith("neuralops.compression")

        is_attention = (
            not is_compressed
            and not is_from_compression
            and (
                cls_name in _WRAPPER_CLASS_NAMES
                or getattr(sub, "is_custom_attention", False)
                or isinstance(sub, BaseAttention)
                or any(isinstance(sub, c) or sub.__class__ == c for c in registered_attn_classes)
            )
        )
        if not is_attention and not is_compressed and not is_from_compression:
            # Also detect generic wrappers holding a registered/BaseAttention child via .attn or ._wrapped_attention
            inner_probe = getattr(sub, "_wrapped_attention", None) or getattr(sub, "attn", None)
            if isinstance(inner_probe, nn.Module) and (
                isinstance(inner_probe, BaseAttention)
                or any(isinstance(inner_probe, c) or inner_probe.__class__ == c for c in registered_attn_classes)
            ):
                is_attention = True

        if is_attention:
            cataloged_attn_names.append(name)
            effective_sub, is_wrapper = _unwrap_attention_module(sub)
            eff_cls_name = effective_sub.__class__.__name__

            target_key = getattr(effective_sub, "target_attention", None)
            if not target_key:
                for reg_k, reg_cls in ATTENTION_REGISTRY.items():
                    if isinstance(reg_cls, type) and (
                        reg_cls == effective_sub.__class__ or isinstance(effective_sub, reg_cls)
                    ):
                        target_key = reg_k
                        break
            if not target_key:
                target_key = eff_cls_name.lower()

            embed_dim = getattr(
                sub,
                "embed_dim",
                getattr(effective_sub, "embed_dim", getattr(effective_sub, "hidden_size", None)),
            )
            if embed_dim is None:
                for p in effective_sub.parameters():
                    if p.ndim == 2:
                        embed_dim = p.shape[0]
                        break
                    if p.ndim == 1:
                        embed_dim = p.shape[0]
            if embed_dim is None:
                embed_dim = 768

            num_heads = getattr(
                sub,
                "num_heads",
                getattr(effective_sub, "num_heads", getattr(effective_sub, "num_attention_heads", 12)),
            )
            core_num_kv_heads = getattr(
                effective_sub,
                "num_kv_heads",
                getattr(effective_sub, "num_key_value_heads", None),
            )
            num_kv_heads = getattr(
                sub,
                "num_kv_heads",
                core_num_kv_heads,
            )
            head_dim = getattr(
                sub,
                "head_dim",
                getattr(effective_sub, "head_dim", embed_dim // max(int(num_heads), 1)),
            )
            rank = getattr(effective_sub, "rank", getattr(effective_sub, "svd_rank", 32))

            sub_info: Dict[str, Any] = {
                "category": "attention",
                "class_name": eff_cls_name,
                "module_import_path": effective_sub.__class__.__module__,
                "target_attention": target_key,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "core_num_kv_heads": core_num_kv_heads,
                "head_dim": head_dim,
                "rank": rank,
                "layer_idx": getattr(effective_sub, "layer_idx", getattr(sub, "layer_idx", None)),
                "is_nested_encoder": getattr(
                    effective_sub, "is_nested_encoder", getattr(sub, "is_nested_encoder", None)
                ),
                "is_decoder": getattr(effective_sub, "is_decoder", getattr(sub, "is_decoder", None)),
                "is_t5": getattr(sub, "is_t5", getattr(effective_sub, "is_t5", False)),
                "is_wrapped": is_wrapper,
                "return_tuple": getattr(sub, "return_tuple", None) if is_wrapper else None,
                "layer_name": getattr(sub, "layer_name", name) if is_wrapper else None,
                "wrapper_class_name": cls_name if is_wrapper else None,
                "wrapper_import_path": sub.__class__.__module__ if is_wrapper else None,
                "param_shapes": {p_name: list(p.shape) for p_name, p in sub.named_parameters(recurse=False)},
                "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
            }
            if eff_cls_name == "ComposableAttention":
                sub_info["projection"] = getattr(effective_sub, "projection_type", "dense")
                sub_info["positional"] = getattr(effective_sub, "positional_type", "none")
                sub_info["kernel_map"] = getattr(effective_sub, "kernel_map_type", "softmax")
                sub_info["mask"] = getattr(effective_sub, "mask_type", "causal")
                sub_info["engine"] = getattr(effective_sub, "engine", "sdpa")

            manifest["submodules"][name] = sub_info

        elif is_compressed:
            in_f = getattr(sub, "in_features", None)
            out_f = getattr(sub, "out_features", None)
            rank = getattr(sub, "rank", getattr(sub, "current_rank", getattr(sub, "r", None)))

            manifest["submodules"][name] = {
                "category": "compressed_linear",
                "class_name": cls_name,
                "module_import_path": sub.__class__.__module__,
                "in_features": in_f,
                "out_features": out_f,
                "rank": rank,
                "has_lora": is_lora,
                "has_bias": getattr(sub, "bias", None) is not None,
                "param_shapes": {p_name: list(p.shape) for p_name, p in sub.named_parameters(recurse=False)},
                "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
            }
            manifest["non_attention_sites"]["compressed_linear"].append(name)

        elif is_lora:
            base_layer = getattr(sub, "base_layer", None)
            in_f = getattr(sub, "in_features", getattr(base_layer, "in_features", None))
            out_f = getattr(sub, "out_features", getattr(base_layer, "out_features", None))
            rank = getattr(sub, "r", getattr(sub, "rank", None))
            if rank is None and hasattr(sub, "lora_A"):
                la = sub.lora_A
                if isinstance(la, torch.Tensor) and la.ndim == 2:
                    rank = int(la.shape[0])
                elif hasattr(la, "weight") and la.weight.ndim == 2:
                    rank = int(la.weight.shape[0])
            manifest["submodules"][name] = {
                "category": "lora_adapter",
                "class_name": cls_name,
                "module_import_path": sub.__class__.__module__,
                "in_features": in_f,
                "out_features": out_f,
                "rank": rank,
                "scaling": getattr(sub, "scaling", None),
                "param_shapes": {p_name: list(p.shape) for p_name, p in sub.named_parameters(recurse=False)},
                "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
            }
            manifest["non_attention_sites"]["lora_adapter"].append(name)

        elif has_frozen_backbone and not _is_vision_tower_path(name):
            direct_params = list(sub.named_parameters(recurse=False))
            if direct_params and any(p.requires_grad for _, p in direct_params):
                leaf = name.rsplit(".", 1)[-1].lower()
                is_norm = "layernorm" in cls_name.lower() or "rmsnorm" in cls_name.lower() or isinstance(sub, nn.LayerNorm)
                is_head = leaf in _HEAD_LEAF_NAMES or any(
                    name.endswith(f".{h}") or name == h for h in _HEAD_LEAF_NAMES
                )
                if is_norm:
                    manifest["submodules"][name] = {
                        "category": "trained_norm",
                        "class_name": cls_name,
                        "module_import_path": sub.__class__.__module__,
                        "param_shapes": {p_name: list(p.shape) for p_name, p in direct_params},
                        "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
                    }
                    manifest["non_attention_sites"]["trained_norm"].append(name)
                elif is_head:
                    manifest["submodules"][name] = {
                        "category": "trained_head",
                        "class_name": cls_name,
                        "module_import_path": sub.__class__.__module__,
                        "in_features": getattr(sub, "in_features", None),
                        "out_features": getattr(sub, "out_features", None),
                        "param_shapes": {p_name: list(p.shape) for p_name, p in direct_params},
                        "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
                    }
                    manifest["non_attention_sites"]["trained_head"].append(name)

    return manifest


def _is_fallback_delta_tensor(key: str) -> bool:
    """Detect compressed/LoRA/adapter tensors when manifest['submodules'] is incomplete."""
    if ".source_attn." in key:
        return False
    leaf = key.rsplit(".", 1)[-1]
    if leaf in {
        "lora_A",
        "lora_B",
        "lora_a",
        "lora_b",
        "weight_u",
        "weight_v",
        "w_down",
        "w_up",
        "qweight",
        "qzeros",
        "g_idx",
        "pruning_mask",
        "weight_mask",
        "int_weight",
        "zero_point",
        "U_packed",
        "U_q",
        "S_packed",
        "S_q",
        "Vt_packed",
        "Vt_q",
        "S_sparse",
        "kept_idx",
        "keep_idx",
    }:
        return True
    lowered = key.lower()
    return any(tok in lowered for tok in (".lora_a.", ".lora_b.", ".adapter.", ".compressed."))


def save_surgical_delta(
    model: nn.Module,
    save_dir: Union[str, Path],
    *,
    weights_filename: str = "attention_weights.safetensors",
    base_model_id: Optional[str] = None,
    surgeried_prefixes: Optional[Sequence[str]] = None,
    trained_param_names: Optional[Sequence[str]] = None,
    anchor_prefixes: Optional[Sequence[str]] = None,
    operations: Optional[list[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    half_precision: bool = True,
    remove_full_weights: bool = True,
) -> Dict[str, Any]:
    """Save strictly the surgical delta parameters/buffers and enriched manifest.

    Excludes frozen backbone tensors, anchor layers, and frozen vision towers
    unless explicitly requested via ``surgeried_prefixes`` or ``trained_param_names``.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    manifest = extract_surgical_manifest(model)
    explicit_surgeried = [str(p).rstrip(".") for p in (surgeried_prefixes or []) if p]
    anchors = [str(p).rstrip(".") for p in (anchor_prefixes or []) if p]
    trained_names = [str(p) for p in (trained_param_names or []) if p]
    trained_set = set(trained_names)

    manifest["schema_version"] = "neuralops.surgery/v2"
    manifest["base_model"] = base_model_id
    manifest["weights_file"] = weights_filename
    manifest["surgeried_prefixes"] = explicit_surgeried
    manifest["anchor_prefixes"] = anchors
    manifest["trained_param_names"] = trained_names
    manifest["operations"] = list(operations) if operations is not None else []
    manifest["metadata"] = dict(metadata) if metadata is not None else {}

    submodule_prefixes = list(manifest.get("submodules", {}).keys())

    delta_state: Dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if ".source_attn." in key:
            continue

        in_trained = key in trained_set
        in_explicit_surgeried = _matches_any_prefix(key, explicit_surgeried)
        in_anchor = _matches_any_prefix(key, anchors)

        if in_anchor and not in_trained:
            continue

        in_vision = _is_vision_tower_path(key)
        if in_vision and not (in_explicit_surgeried or in_trained):
            continue

        include = (
            in_trained
            or in_explicit_surgeried
            or _matches_any_prefix(key, submodule_prefixes)
            or _is_fallback_delta_tensor(key)
        )
        if not include:
            continue

        t = tensor.detach().cpu()
        if half_precision and t.is_floating_point():
            t = t.half()
        delta_state[key] = t.contiguous().clone()

    from safetensors.torch import save_file

    weights_path = path / weights_filename
    tmp_weights = path / f".{weights_filename}.tmp.{os.getpid()}"
    save_file(delta_state, str(tmp_weights))
    os.replace(tmp_weights, weights_path)

    manifest_path = path / "surgical_manifest.json"
    tmp_manifest = path / f".surgical_manifest.json.tmp.{os.getpid()}"
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    if remove_full_weights:
        for heavy_name in ("pytorch_model.bin", "model.safetensors", "surgical_model_state.pt", "optimizer.pt"):
            if heavy_name == weights_filename:
                continue
            heavy_path = path / heavy_name
            if heavy_path.exists():
                heavy_path.unlink()

    saved_bytes = weights_path.stat().st_size if weights_path.exists() else 0
    if not delta_state:
        logger.warning("save_surgical_delta wrote 0 delta tensors to %s", weights_path)
    else:
        logger.info(
            "Saved surgical delta (%d tensors, %.2f MB) to %s",
            len(delta_state),
            saved_bytes / (1024 * 1024),
            weights_path,
        )

    return {
        "weights_file": weights_filename,
        "saved_tensor_count": len(delta_state),
        "saved_bytes": saved_bytes,
        "manifest": manifest,
    }


def save_surgical_checkpoint(
    model: nn.Module,
    save_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save surgical manifest and full state dict to a clean directory."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    manifest = extract_surgical_manifest(model)
    if metadata is not None:
        manifest["user_metadata"] = metadata

    manifest_file = path / "surgical_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    state_dict = model.state_dict()
    clean_state_dict = {k: v for k, v in state_dict.items() if ".source_attn." not in k}

    try:
        from safetensors.torch import save_file

        clean_state = {k: v.contiguous().clone() for k, v in clean_state_dict.items()}
        save_file(clean_state, str(path / "model.safetensors"))
    except (ImportError, RuntimeError, OSError) as e_st:
        logger.debug("Safetensors save skipped: %s", e_st)

    torch.save(clean_state_dict, str(path / "surgical_model_state.pt"))
    logger.info("Saved surgical checkpoint with %d modified submodules to %s", len(manifest["submodules"]), path)
    return path


def _load_state_file(file_path: Path, target_device: Any) -> Dict[str, torch.Tensor]:
    if file_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(file_path), device=str(target_device or "cpu"))
    return torch.load(str(file_path), map_location=target_device, weights_only=True)


def load_surgical_checkpoint(
    base_model: nn.Module,
    load_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Reconstruct surgical modules inside base_model and load saved weights."""
    path = Path(load_dir)
    manifest_file = path / "surgical_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing surgical_manifest.json in {load_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    submodules_info = manifest.get("submodules", {})
    from neuralops.core.model_utils import replace_module

    target_device = map_location
    if target_device is None:
        try:
            target_device = next(base_model.parameters()).device
        except (StopIteration, RuntimeError):
            target_device = torch.device("cpu")

    for module_path, info in submodules_info.items():
        category = info.get("category")
        if category in ("trained_head", "trained_norm"):
            # Native model modules whose trained weights are restored by load_state_dict.
            continue

        cls_name = info.get("class_name")
        import_path = info.get("module_import_path")

        if category == "lora_adapter":
            try:
                existing = base_model.get_submodule(module_path)
            except AttributeError:
                existing = None
            if existing is not None and hasattr(existing, "lora_A") and hasattr(existing, "lora_B"):
                continue

            rank = int(info.get("rank") or 8)
            cls = None
            if import_path and cls_name:
                try:
                    mod = importlib.import_module(import_path)
                    cls = getattr(mod, cls_name)
                except (ImportError, AttributeError, ModuleNotFoundError):
                    cls = None

            if existing is not None:
                if cls is not None:
                    try:
                        new_sub = cls(existing, r=rank)
                    except TypeError:
                        new_sub = cls(existing)
                elif "lora_A.weight" in (info.get("param_shapes") or {}) and isinstance(existing, nn.Linear):
                    from neuralops.trainer.methods.lora import LoRALinear

                    new_sub = LoRALinear(existing, r=rank)
                else:
                    from neuralops.hub.surgery_loader import _LoRALinearAdapter

                    scaling_val = info.get("scaling")
                    alpha_val = float(scaling_val) * rank if scaling_val is not None else 16.0
                    new_sub = _LoRALinearAdapter(existing, r=rank, alpha=alpha_val)
                if target_device is not None:
                    new_sub = new_sub.to(target_device)
                replace_module(base_model, module_path, new_sub)
            continue

        try:
            mod = importlib.import_module(import_path)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            from neuralops.core.registry import ATTENTION_REGISTRY

            target_key = info.get("target_attention", (cls_name or "").lower())
            cls = ATTENTION_REGISTRY.get(target_key, unwrap=True) or ATTENTION_REGISTRY.get(
                (cls_name or "").lower(), unwrap=True
            )
            if cls is None:
                raise ImportError(f"Could not import or resolve class '{cls_name}' from '{import_path}'")
        assert callable(cls), f"Class {cls_name} is not callable"

        if category == "attention":
            if cls_name == "ComposableAttention":
                new_sub = cls(
                    embed_dim=info.get("embed_dim", 768),
                    num_heads=info.get("num_heads", 12),
                    projection=info.get("projection", "dense"),
                    positional=info.get("positional", "none"),
                    kernel_map=info.get("kernel_map", "softmax"),
                    mask=info.get("mask", "causal"),
                    engine=info.get("engine", "sdpa"),
                )
            else:
                embed_dim = info.get("embed_dim", 768)
                num_heads = info.get("num_heads", 12)
                num_kv_heads = info.get("core_num_kv_heads", info.get("num_kv_heads", None))
                rank = info.get("rank", 32)
                from transformers import BertConfig

                cfg = BertConfig(
                    hidden_size=embed_dim,
                    num_attention_heads=num_heads,
                    rank=rank,
                    svd_rank=rank,
                )
                if num_kv_heads is not None:
                    cfg.num_key_value_heads = num_kv_heads
                    cfg.num_kv_heads = num_kv_heads

                init_kwargs: Dict[str, Any] = {
                    "rank": rank,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "d_model": embed_dim,
                }
                if num_kv_heads is not None:
                    init_kwargs["num_kv_heads"] = num_kv_heads
                    init_kwargs["num_key_value_heads"] = num_kv_heads

                try:
                    new_sub = cls(config=cfg, **init_kwargs)
                except (TypeError, ValueError, AttributeError):
                    try:
                        new_sub = cls(config=cfg)
                    except (TypeError, ValueError, AttributeError):
                        try:
                            new_sub = cls(**init_kwargs)
                        except (TypeError, ValueError, AttributeError):
                            new_sub = cls()

                if info.get("is_nested_encoder") is not None:
                    new_sub.is_nested_encoder = bool(info["is_nested_encoder"])
                if info.get("is_decoder") is not None:
                    new_sub.is_decoder = bool(info["is_decoder"])
                if info.get("layer_idx") is not None:
                    new_sub.layer_idx = info["layer_idx"]

            wrapper_cls_name = info.get("wrapper_class_name")
            if wrapper_cls_name == "QwenAttentionAdapter":
                from neuralops.layers.qwen_adapter import QwenAttentionAdapter

                embed_dim = info.get("embed_dim", 896)
                num_heads = info.get("num_heads", 14)
                num_kv_heads = info.get("num_kv_heads") or num_heads
                head_dim = info.get("head_dim") or (embed_dim // max(int(num_heads), 1))
                new_sub = QwenAttentionAdapter(
                    core_attention=new_sub,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_causal=True,
                    bypass_internal_out_proj=True,
                )
            elif wrapper_cls_name == "BertAttentionAdapter":
                from neuralops.layers.bert_adapter import BertAttentionAdapter

                new_sub = BertAttentionAdapter(
                    core_attention=new_sub,
                    embed_dim=info.get("embed_dim", 768),
                    num_heads=info.get("num_heads", 12),
                    bypass_internal_out_proj=True,
                )
            elif info.get("is_wrapped", False) or wrapper_cls_name == "UniversalAttentionWrapper":
                from neuralops.utils.replace_attention import UniversalAttentionWrapper

                wrapped_sub = UniversalAttentionWrapper(
                    new_sub,
                    layer_name=info.get("layer_name", module_path),
                    return_tuple=info.get("return_tuple", None),
                )
                if info.get("is_t5") is not None:
                    wrapped_sub.is_t5 = bool(info["is_t5"])
                new_sub = wrapped_sub

            if target_device is not None:
                new_sub = new_sub.to(target_device)
            replace_module(base_model, module_path, new_sub)

        elif category == "compressed_linear":
            in_f = info.get("in_features") or 768
            out_f = info.get("out_features") or 768
            rank = info.get("rank") or 32
            has_b = info.get("has_bias", True)
            dummy_linear = nn.Linear(in_f, out_f, bias=has_b)

            new_sub = None
            for ctor in (
                lambda: cls(in_features=in_f, out_features=out_f, rank=rank, bias=has_b),
                lambda: cls.from_linear(dummy_linear, rank=rank),
                lambda: cls(dummy_linear, rank=rank),
                lambda: cls(dummy_linear),
                lambda: cls(in_features=in_f, out_features=out_f),
            ):
                try:
                    new_sub = ctor()
                    break
                except (TypeError, ValueError, AttributeError):
                    continue
            if new_sub is None:
                raise RuntimeError(f"Could not reconstruct compressed_linear '{cls_name}' at '{module_path}'")

            if target_device is not None:
                new_sub = new_sub.to(target_device)
            replace_module(base_model, module_path, new_sub)

    candidate_names: list[str] = []
    manifest_weights = manifest.get("weights_file")
    if isinstance(manifest_weights, str) and manifest_weights:
        candidate_names.append(manifest_weights)
    for fallback_name in (
        "attention_weights.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "surgical_model_state.pt",
    ):
        if fallback_name not in candidate_names:
            candidate_names.append(fallback_name)

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    loaded_file: Optional[Path] = None
    last_err: Optional[Exception] = None
    for fname in candidate_names:
        cand = path / fname
        if not cand.exists():
            continue
        try:
            state_dict = _load_state_file(cand, target_device)
            loaded_file = cand
            break
        except (ImportError, RuntimeError, OSError) as exc:
            last_err = exc
            logger.warning("Failed to load checkpoint file %s: %s", cand, exc)

    if state_dict is None or loaded_file is None:
        if last_err is not None:
            raise RuntimeError(f"Failed loading weights from {load_dir}: {last_err}") from last_err
        raise FileNotFoundError(f"No valid state dict found in {load_dir}")

    is_delta_ckpt = (
        manifest.get("schema_version") == "neuralops.surgery/v2"
        or loaded_file.name == "attention_weights.safetensors"
    )
    effective_strict = False if is_delta_ckpt else strict
    load_res = base_model.load_state_dict(state_dict, strict=effective_strict)
    if is_delta_ckpt and strict and getattr(load_res, "unexpected_keys", None):
        logger.warning("Unexpected keys in delta checkpoint %s: %s", loaded_file, load_res.unexpected_keys)

    if target_device is not None:
        base_model.to(device=target_device)
    base_model.eval()

    logger.info("Successfully loaded surgical model checkpoint from %s (%s)", path, loaded_file.name)
    return base_model, manifest


__all__ = [
    "extract_surgical_manifest",
    "save_surgical_delta",
    "save_surgical_checkpoint",
    "load_surgical_checkpoint",
]
