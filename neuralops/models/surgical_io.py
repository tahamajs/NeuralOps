"""
Surgical Model IO and Checkpointing

Model state serialization and weight conversion routines for seamless HuggingFace checkpoint interoperability.

Author: Taha Majlesi
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_surgical_manifest(model: nn.Module) -> Dict[str, Any]:
    """Walk model tree and extract exact metadata for all surgical modules.

    Captures:
      - Category: 'attention' vs 'compressed_linear'
      - Exact hierarchical module_path
      - Module class, registration key, constructor parameters, and parameter shapes
    """
    manifest: Dict[str, Any] = {
        "neuralops_version": "1.3.7",
        "model_class": model.__class__.__name__,
        "submodules": {},
    }

    # Set of known compressor linear class names
    known_compressed_linears = {
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

    from neuralops.core.registry import ATTENTION_REGISTRY
    registered_attn_classes = tuple(ATTENTION_REGISTRY.values())
    cataloged_attn_names: list[str] = []

    for name, sub in model.named_modules():
        if not name:
            continue
        cls_name = sub.__class__.__name__

        # Skip submodules that belong inside an already-cataloged attention block
        if any(name.startswith(p + ".") for p in cataloged_attn_names):
            continue

        # Check if Attention module (AttentionWrapper, ComposableAttention, or registered attention)
        is_attention = (
            cls_name in ("AttentionWrapper", "ComposableAttention", "ReplaceAttention", "UniversalAttentionWrapper")
            or getattr(sub, "is_custom_attention", False)
            or any(isinstance(sub, c) or sub.__class__ == c for c in registered_attn_classes)
        )

        is_compressed = (
            cls_name in known_compressed_linears
            or getattr(sub, "is_custom_linear", False)
            or (hasattr(sub, "U") and hasattr(sub, "V"))
            or (hasattr(sub, "w_down") and hasattr(sub, "w_up"))
        )

        if is_attention:
            cataloged_attn_names.append(name)
            # If sub is a wrapper, inspect its inner attention module
            is_wrapper = hasattr(sub, "_wrapped_attention") and getattr(sub, "_wrapped_attention") is not None
            effective_sub = sub._wrapped_attention if is_wrapper else sub
            eff_cls_name = effective_sub.__class__.__name__

            # Resolve registered key if available
            target_key = getattr(effective_sub, "target_attention", None)
            if not target_key:
                for reg_k, reg_cls in ATTENTION_REGISTRY.items():
                    if reg_cls == effective_sub.__class__ or isinstance(effective_sub, reg_cls):
                        target_key = reg_k
                        break
            if not target_key:
                target_key = eff_cls_name.lower()

            # Determine embed_dim, num_heads, rank accurately
            embed_dim = getattr(effective_sub, "embed_dim", getattr(effective_sub, "hidden_size", None))
            if embed_dim is None:
                for p in effective_sub.parameters():
                    if p.ndim == 2:
                        embed_dim = p.shape[0]
                        break
                    elif p.ndim == 1:
                        embed_dim = p.shape[0]
            if embed_dim is None:
                embed_dim = 768

            num_heads = getattr(effective_sub, "num_heads", getattr(effective_sub, "num_attention_heads", 12))
            num_kv_heads = getattr(effective_sub, "num_kv_heads", getattr(effective_sub, "num_key_value_heads", None))
            rank = getattr(effective_sub, "rank", getattr(effective_sub, "svd_rank", 32))

            sub_info: Dict[str, Any] = {
                "category": "attention",
                "class_name": eff_cls_name,
                "module_import_path": effective_sub.__class__.__module__,
                "target_attention": target_key,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "rank": rank,
                "layer_idx": getattr(effective_sub, "layer_idx", getattr(sub, "layer_idx", None)),
                "is_nested_encoder": getattr(effective_sub, "is_nested_encoder", getattr(sub, "is_nested_encoder", None)),
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
            rank = getattr(sub, "rank", getattr(sub, "current_rank", None))

            manifest["submodules"][name] = {
                "category": "compressed_linear",
                "class_name": cls_name,
                "module_import_path": sub.__class__.__module__,
                "in_features": in_f,
                "out_features": out_f,
                "rank": rank,
                "has_bias": getattr(sub, "bias", None) is not None,
                "param_shapes": {p_name: list(p.shape) for p_name, p in sub.named_parameters(recurse=False)},
                "buffer_shapes": {b_name: list(b.shape) for b_name, b in sub.named_buffers(recurse=False)},
            }

    return manifest


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
    # Filter out transient source_attn proxies that were attached only for weight transplantation
    clean_state_dict = {
        k: v for k, v in state_dict.items() if ".source_attn." not in k
    }

    # Try saving with safetensors if available
    saved_safetensors = False
    try:
        from safetensors.torch import save_file
        # Clone tensors to ensure contiguous unshared storage
        clean_state = {k: v.contiguous().clone() for k, v in clean_state_dict.items()}
        save_file(clean_state, str(path / "model.safetensors"))
        saved_safetensors = True
    except Exception as e_st:
        logger.debug("Safetensors save skipped: %s", e_st)

    # Always write standard PyTorch checkpoint as reliable companion
    torch.save(clean_state_dict, str(path / "surgical_model_state.pt"))
    logger.info("Saved surgical checkpoint with %d modified submodules to %s", len(manifest["submodules"]), path)
    return path


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

    # Infer base_model device if map_location is not explicitly passed
    target_device = map_location
    if target_device is None:
        try:
            target_device = next(base_model.parameters()).device
        except (StopIteration, Exception):
            target_device = torch.device("cpu")

    # 1. Reconstruct all modified submodules
    for module_path, info in submodules_info.items():
        category = info.get("category")
        cls_name = info.get("class_name")
        import_path = info.get("module_import_path")

        # Import class dynamically or resolve from ATTENTION_REGISTRY
        try:
            mod = importlib.import_module(import_path)
            cls = getattr(mod, cls_name)
        except Exception:
            from neuralops.core.registry import ATTENTION_REGISTRY
            target_key = info.get("target_attention", cls_name.lower())
            cls = ATTENTION_REGISTRY.get(target_key, unwrap=True) or ATTENTION_REGISTRY.get(cls_name.lower(), unwrap=True)
            if cls is None:
                raise ImportError(f"Could not import or resolve class '{cls_name}' from '{import_path}'")

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
                num_kv_heads = info.get("num_kv_heads", None)
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

                init_kwargs = {
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
                except Exception:
                    try:
                        new_sub = cls(config=cfg)
                    except Exception:
                        try:
                            new_sub = cls(**init_kwargs)
                        except Exception:
                            new_sub = cls()

                if info.get("is_nested_encoder") is not None:
                    new_sub.is_nested_encoder = bool(info["is_nested_encoder"])
                if info.get("is_decoder") is not None:
                    new_sub.is_decoder = bool(info["is_decoder"])
                if info.get("layer_idx") is not None:
                    new_sub.layer_idx = info["layer_idx"]

            if info.get("is_wrapped", False) or info.get("wrapper_class_name") == "UniversalAttentionWrapper":
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
            in_f = info.get("in_features", 768)
            out_f = info.get("out_features", 768)
            rank = info.get("rank", 32)
            has_b = info.get("has_bias", True)

            # Reconstruct factorized linear
            try:
                new_sub = cls(in_features=in_f, out_features=out_f, rank=rank, bias=has_b)
            except Exception:
                try:
                    dummy_linear = nn.Linear(in_f, out_f, bias=has_b)
                    new_sub = cls.from_linear(dummy_linear, rank=rank)
                except Exception:
                    new_sub = cls(in_features=in_f, out_features=out_f)

            if target_device is not None:
                new_sub = new_sub.to(target_device)
            replace_module(base_model, module_path, new_sub)

    # 2. Load Weights into reconstructed model
    pt_file = path / "surgical_model_state.pt"
    st_file = path / "model.safetensors"

    if st_file.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(st_file), device=str(target_device or "cpu"))
        except Exception:
            state_dict = torch.load(str(pt_file), map_location=target_device, weights_only=True)
    elif pt_file.exists():
        state_dict = torch.load(str(pt_file), map_location=target_device, weights_only=True)
    else:
        raise FileNotFoundError(f"No valid state dict found in {load_dir}")

    base_model.load_state_dict(state_dict, strict=strict)
    if target_device is not None:
        base_model.to(device=target_device)
    base_model.eval()

    logger.info("Successfully loaded surgical model checkpoint from %s", path)
    return base_model, manifest


__all__ = [
    "extract_surgical_manifest",
    "save_surgical_checkpoint",
    "load_surgical_checkpoint",
]
