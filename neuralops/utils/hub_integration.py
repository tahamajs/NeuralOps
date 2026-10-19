#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Selective Hugging Face Hub serialization for NeuralOps attention surgery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _attention_prefixes(model: torch.nn.Module) -> set[str]:
    """Return state-dict prefixes belonging to injected attention modules."""
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        module_name = module.__class__.__module__
        class_name = module.__class__.__name__.lower()
        if "neuralops" in module_name or "attention" in class_name or "lora" in class_name:
            if name:
                prefixes.add(name + ".")
    return prefixes


def _select_attention_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract only injected attention and adapter tensors from ``model``."""
    prefixes = _attention_prefixes(model)
    selected: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        is_adapter = "lora_" in key.lower() or "pissa" in key.lower() or "dora" in key.lower()
        if is_adapter or any(key.startswith(prefix) for prefix in prefixes):
            selected[key] = value.detach().cpu().contiguous()
    if not selected:
        raise ValueError("No NeuralOps attention or adapter weights were found in the model.")
    return selected


def _attention_config(model: torch.nn.Module) -> dict[str, Any]:
    """Build a portable manifest for reconstructing a surgically edited model."""
    config = getattr(model, "config", None)
    rank = None
    for module in model.modules():
        if "lora" not in module.__class__.__name__.lower():
            continue
        value = getattr(module, "r", None)
        if value is None:
            value = getattr(module, "rank", None)
        if value is not None:
            rank = int(value)
            break
    metadata = getattr(model, "_neuralops_metadata", None)
    if isinstance(metadata, list):
        metadata = metadata[-1] if metadata else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "base_model_name": getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None),
        "attention_type": getattr(config, "neuralops_attention", None)
        or metadata.get("attention_type")
        or metadata.get("type")
        or metadata.get("kwargs", {}).get("attention_type"),
        "lora_rank": rank,
        "format": "neuralops_attention_only_v1",
    }


def save_attention_only(
    model: torch.nn.Module,
    save_directory: str | os.PathLike[str],
    *,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
) -> str:
    """Save injected attention/adapter weights and a reconstruction manifest.

    The base model is never serialized. ``save_directory`` contains only
    ``attention_weights.safetensors`` and ``config.json``; when requested the
    same two files are uploaded to the Hub.
    """
    target = Path(save_directory)
    target.mkdir(parents=True, exist_ok=True)
    state = _select_attention_state(model)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Selective saving requires safetensors.") from exc
    save_file(state, str(target / "attention_weights.safetensors"))
    manifest = _attention_config(model)
    (target / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id is required when push_to_hub=True.")
        from huggingface_hub import HfApi

        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(target),
            repo_id=repo_id,
            allow_patterns=["attention_weights.safetensors", "config.json"],
            commit_message="Upload NeuralOps attention-only weights",
        )
    return str(target)


def load_attention_from_hub(
    base_model_name: str,
    repo_id: str | os.PathLike[str],
    *,
    device: str | torch.device | None = None,
    **from_pretrained_kwargs: Any,
) -> torch.nn.Module:
    """Load a base HF model, repeat surgery from the manifest, and load weights."""
    source = Path(repo_id)
    if source.is_dir():
        manifest_path = source / "config.json"
        weights_path = source / "attention_weights.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        manifest_path = Path(hf_hub_download(str(repo_id), "config.json"))
        weights_path = Path(hf_hub_download(str(repo_id), "attention_weights.safetensors"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attention_type = manifest.get("attention_type")
    if not attention_type:
        raise ValueError("Selective checkpoint is missing config.json:attention_type.")
    from transformers import AutoModel
    from neuralops.api import replace_attention

    model = AutoModel.from_pretrained(base_model_name, **from_pretrained_kwargs)
    lora_rank = manifest.get("lora_rank")
    model = replace_attention(
        model,
        attention_type,
        copy_weights=True,
        inject_lora=lora_rank is not None,
        lora_rank=int(lora_rank or 8),
    )
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if device is not None:
        model = model.to(device)
    model._neuralops_load_report = {"missing_keys": missing, "unexpected_keys": unexpected}
    return model


def attach_attention_methods(model: torch.nn.Module) -> torch.nn.Module:
    """Attach the notebook-friendly selective-save method to a model instance."""
    from types import MethodType

    model.save_attention_only = MethodType(save_attention_only, model)
    return model
