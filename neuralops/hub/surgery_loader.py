#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Public Hub Loader & Decoder Attention Surgery Utilities.

Provides a single-call public loader ``load_surgeried_model`` capable of
reconstructing any NeuralOps-surgeried BERT, Qwen2, TinyLlama/LLaMA, or
LLaVA vision-language checkpoint from delta-only safetensors (or legacy
``pytorch_model.bin``) and frozen Hugging Face base models.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _LoRALinearAdapter(nn.Module):
    """Lightweight LoRA linear wrapper matching SimpleLoRALinear state_dict keys."""

    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_f = getattr(base_layer, "in_features", None)
        out_f = getattr(base_layer, "out_features", None)
        if in_f is None or out_f is None:
            for p in base_layer.parameters():
                if p.ndim == 2:
                    out_f, in_f = p.shape
                    break

        self.r = int(r)
        self.scaling = float(alpha) / max(self.r, 1)
        if in_f and out_f:
            ref_param = next(base_layer.parameters(), None)
            dtype = ref_param.dtype if ref_param is not None else torch.float32
            device = ref_param.device if ref_param is not None else torch.device("cpu")
            self.lora_A = nn.Parameter(
                torch.randn(self.r, in_f, dtype=dtype, device=device) * (1.0 / math.sqrt(max(self.r, 1)))
            )
            self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, dtype=dtype, device=device))
            self.has_lora = True
        else:
            self.has_lora = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base_layer(x)
        if self.has_lora:
            dtype = x.dtype
            lora_out = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scaling
            res = res + lora_out.to(dtype)
        return res


def normalize_base_model_id(model_name: str) -> str:
    """Normalize local cache paths or aliases to canonical Hugging Face Hub model IDs."""
    cleaned = (model_name or "").strip()
    if not cleaned:
        return cleaned

    lower = cleaned.lower()
    if "llava-interleave-qwen-0.5b" in lower or ("llava" in lower and "0.5b" in lower):
        return "llava-hf/llava-interleave-qwen-0.5b-hf"
    if "qwen2.5-0.5b-instruct" in lower or "qwen_2.5_0.5b_instruct" in lower:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    if "qwen2.5-0.5b" in lower or "qwen_2.5_0.5b" in lower:
        return "Qwen/Qwen2.5-0.5B"
    if "tinyllama" in lower and ("1.1b" in lower or cleaned.startswith("/") or os.path.exists(cleaned)):
        return "TinyLlama/TinyLlama_v1.1"
    if "bert-base-uncased" in lower or "bert_base_uncased" in lower or "bert_sst2" in lower:
        if "/" in cleaned and not cleaned.startswith("/") and not os.path.exists(cleaned):
            return cleaned
        return "bert-base-uncased"

    if cleaned.startswith("/") or os.path.exists(cleaned):
        tail = Path(cleaned.rstrip("/")).name
        tail_lower = tail.lower()
        if tail_lower.startswith("qwen"):
            return f"Qwen/{tail}"
        if tail_lower.startswith("tinyllama"):
            return "TinyLlama/TinyLlama_v1.1" if "1.1" in tail_lower else f"TinyLlama/{tail}"
        if "llava" in tail_lower:
            return f"llava-hf/{tail}"
        if "bert" in tail_lower:
            return "bert-base-uncased"
        return tail

    return cleaned


def _locate_decoder_layers(model: nn.Module) -> Tuple[nn.ModuleList, str, Any]:
    """Locate decoder layer list, dotted prefix, and text config across HF decoder/VLM models."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None

    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers, "language_model.model.layers", text_cfg
        if hasattr(lm, "layers"):
            return lm.layers, "language_model.layers", text_cfg

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.layers", text_cfg

    if hasattr(model, "layers"):
        return model.layers, "layers", text_cfg

    raise ValueError(
        f"Cannot locate decoder transformer layers in {model.__class__.__name__} "
        "(expected .language_model.model.layers, .model.layers, or .layers)."
    )


def _instantiate_core_attention(
    attn_key: str,
    hidden_size: int,
    num_heads: int,
    text_cfg: Any,
    rank: Optional[int],
    layer_idx: int,
) -> nn.Module:
    """Instantiate a registered NeuralOps attention module with graceful kwarg fallback."""
    from neuralops.core.registry import create_attention

    attempts: list[dict[str, Any]] = []
    if rank is not None:
        attempts.extend(
            [
                {
                    "dim": hidden_size,
                    "num_heads": num_heads,
                    "rank": rank,
                    "config": text_cfg,
                    "is_causal": True,
                    "layer_idx": layer_idx,
                },
                {
                    "dim": hidden_size,
                    "num_heads": num_heads,
                    "rank": rank,
                    "config": text_cfg,
                    "is_causal": True,
                },
            ]
        )
    attempts.extend(
        [
            {
                "dim": hidden_size,
                "num_heads": num_heads,
                "config": text_cfg,
                "is_causal": True,
                "layer_idx": layer_idx,
            },
            {
                "dim": hidden_size,
                "num_heads": num_heads,
                "config": text_cfg,
                "is_causal": True,
            },
            {
                "dim": hidden_size,
                "num_heads": num_heads,
                "config": text_cfg,
            },
        ]
    )
    last_err: Optional[Exception] = None
    for kw in attempts:
        try:
            return create_attention(attn_key, **kw)
        except TypeError as exc:
            last_err = exc
            continue
    raise TypeError(f"Failed to instantiate attention '{attn_key}': {last_err}")


def apply_decoder_attention_surgery(
    model: nn.Module,
    attn_key: str,
    pattern: str = "hybrid_3_1",
    rank: Optional[int] = 64,
    dtype: Any = None,
) -> Tuple[nn.Module, list[str], list[str]]:
    """Transplant a NeuralOps attention mechanism into decoder layers via QwenAttentionAdapter.

    Supports Qwen2, TinyLlama/LLaMA, and LLaVA (``.language_model.model.layers`` or
    ``.model.layers``).

    Args:
        model: Loaded Hugging Face decoder or VLM model.
        attn_key: Registered key in ``ATTENTION_REGISTRY``.
        pattern: ``"hybrid_3_1"`` (retain native attention every 4th layer),
            ``"hybrid_1_1"`` (every 2nd layer), or ``"all"``.
        rank: Rank parameter forwarded to rank-aware attention constructors.
        dtype: Optional target dtype for newly instantiated adapters.

    Returns:
        ``(model, target_prefixes, anchor_prefixes)``
    """
    from neuralops.core.registry import ATTENTION_REGISTRY
    from neuralops.layers.qwen_adapter import QwenAttentionAdapter

    if attn_key not in ATTENTION_REGISTRY:
        raise KeyError(f"Attention key '{attn_key}' not found in ATTENTION_REGISTRY.")

    layers, layers_prefix, text_cfg = _locate_decoder_layers(model)
    hidden_size = getattr(text_cfg, "hidden_size", 896)
    num_heads = getattr(text_cfg, "num_attention_heads", 14)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(text_cfg, "head_dim", hidden_size // max(int(num_heads), 1))

    target_prefixes: list[str] = []
    anchor_prefixes: list[str] = []

    for idx, layer in enumerate(layers):
        attn_path = f"{layers_prefix}.{idx}.self_attn"
        if pattern == "hybrid_3_1" and (idx % 4 == 3):
            anchor_prefixes.append(attn_path)
            continue
        if pattern == "hybrid_1_1" and (idx % 2 == 1):
            anchor_prefixes.append(attn_path)
            continue

        orig_attn = getattr(layer, "self_attn", None)
        if orig_attn is None:
            continue

        ref_param = next(orig_attn.parameters(), None)
        target_device = ref_param.device if ref_param is not None else torch.device("cpu")
        target_dtype = dtype if dtype is not None else (ref_param.dtype if ref_param is not None else None)

        core_attn = _instantiate_core_attention(
            attn_key=attn_key,
            hidden_size=hidden_size,
            num_heads=num_heads,
            text_cfg=text_cfg,
            rank=rank,
            layer_idx=idx,
        )

        adapter = QwenAttentionAdapter(
            core_attention=core_attn,
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            inherit_weights_from=orig_attn,
            is_causal=True,
            bypass_internal_out_proj=True,
        )
        if target_dtype is not None:
            adapter = adapter.to(device=target_device, dtype=target_dtype)
        else:
            adapter = adapter.to(device=target_device)

        layer.self_attn = adapter
        target_prefixes.append(attn_path)

    logger.info(
        "Decoder attention surgery complete: transplanted %d/%d layers with '%s' (pattern=%s, anchors=%d)",
        len(target_prefixes),
        len(layers),
        attn_key,
        pattern,
        len(anchor_prefixes),
    )
    return model, target_prefixes, anchor_prefixes


def _resolve_repo_file(
    repo_id: str,
    filename: str,
    *,
    optional: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a file from a local directory or download it from Hugging Face Hub."""
    local_dir = Path(repo_id)
    if local_dir.is_dir():
        cand = local_dir / filename
        if cand.exists():
            return cand
        if optional:
            return None
        raise FileNotFoundError(f"Missing {filename} in local checkpoint directory {repo_id}")

    from huggingface_hub import hf_hub_download
    from neuralops.hub.uploader import retry_hf_api_call

    resolved_token = token or os.environ.get("HF_TOKEN")
    try:
        downloaded = retry_hf_api_call(
            hf_hub_download,
            repo_id=str(repo_id),
            filename=filename,
            token=resolved_token,
            revision=revision,
            cache_dir=cache_dir,
        )
        return Path(downloaded)
    except Exception as exc:
        if optional:
            logger.debug("Optional file %s not found in %s: %s", filename, repo_id, exc)
            return None
        raise


def _resolve_dtype(torch_dtype: Any) -> Any:
    if torch_dtype is None or torch_dtype == "auto":
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
        return getattr(torch, torch_dtype)
    return torch_dtype


def _infer_base_model_id(repo_id: str, cfg_dict: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Infer a valid Hugging Face base model ID from config.json and surgical_manifest.json."""
    raw = (
        manifest.get("base_model")
        or cfg_dict.get("base_model_name_or_path")
        or cfg_dict.get("_name_or_path")
        or cfg_dict.get("base_model_name")
        or ""
    )
    normalized = normalize_base_model_id(str(raw)) if raw else ""
    if normalized and normalized != repo_id:
        if "/" in normalized or normalized in {"bert-base-uncased", "bert-large-uncased", "gpt2"}:
            return normalized

    model_type = (cfg_dict.get("model_type") or "").lower()
    architectures = " ".join(cfg_dict.get("architectures") or []).lower()
    if model_type == "llava" or "llava" in architectures:
        return "llava-hf/llava-interleave-qwen-0.5b-hf"
    if model_type == "qwen2" or "qwen2" in architectures:
        return "Qwen/Qwen2.5-0.5B"
    if model_type == "llama" or "llama" in architectures:
        return "TinyLlama/TinyLlama_v1.1"
    if model_type == "bert" or "bert" in architectures:
        return "bert-base-uncased"

    return normalized or str(raw) or repo_id


def _select_model_class(task: Optional[str], cfg_dict: dict[str, Any]) -> Tuple[Any, bool]:
    """Select the appropriate Hugging Face Transformers model class and whether num_labels applies."""
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        BertForSequenceClassification,
    )

    architectures = cfg_dict.get("architectures") or []
    model_type = (cfg_dict.get("model_type") or "").lower()

    if task == "masked-lm":
        return AutoModelForMaskedLM, False
    if task == "sequence-classification":
        if model_type == "bert":
            return BertForSequenceClassification, True
        return AutoModelForSequenceClassification, True
    if task == "token-classification":
        return AutoModelForTokenClassification, True
    if task in ("vision-language", "vlm"):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration, False
    if task == "causal-lm":
        return AutoModelForCausalLM, False
    if task == "feature-extraction":
        return AutoModel, False

    if model_type == "llava" or any("Llava" in a for a in architectures):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration, False
    if any("SequenceClassification" in a for a in architectures):
        if model_type == "bert":
            return BertForSequenceClassification, True
        return AutoModelForSequenceClassification, True
    if any("MaskedLM" in a for a in architectures):
        return AutoModelForMaskedLM, False
    if any("TokenClassification" in a for a in architectures):
        return AutoModelForTokenClassification, True
    if model_type in ("qwen2", "llama", "mistral", "gpt2") or any("CausalLM" in a for a in architectures):
        return AutoModelForCausalLM, False
    if model_type == "bert":
        if "id2label" in cfg_dict or "num_labels" in cfg_dict:
            return BertForSequenceClassification, True
        return AutoModel, False
    return AutoModelForCausalLM, False


def _infer_surgery_spec(
    cfg_dict: dict[str, Any],
    manifest: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Extract surgery type, method key, pattern, rank, and LoRA configuration."""
    meta = manifest.get("metadata") or {}
    ops = manifest.get("operations") or []

    attn_key = (
        cfg_dict.get("attention_type")
        or cfg_dict.get("neuralops_attention_type")
        or cfg_dict.get("neuralops_attention")
        or meta.get("attention_type")
    )
    comp_key = (
        cfg_dict.get("compression_method")
        or cfg_dict.get("compressor_key")
        or cfg_dict.get("compressor")
        or meta.get("compression_method")
        or meta.get("compressor_key")
    )
    pattern = cfg_dict.get("surgery_pattern") or cfg_dict.get("pattern") or meta.get("pattern") or manifest.get("pattern")
    rank = cfg_dict.get("compression_rank") or cfg_dict.get("rank") or meta.get("rank") or manifest.get("rank")
    lora_rank = cfg_dict.get("lora_rank") or meta.get("lora_rank")

    for op in ops:
        if not isinstance(op, dict):
            continue
        op_name = str(op.get("component_type") or op.get("op") or op.get("type") or "").lower()
        if not attn_key and ("attention" in op_name or "replace" in op_name):
            attn_key = op.get("operator_key") or op.get("attn_key") or op.get("attention_type") or op.get("method")
            pattern = pattern or op.get("pattern")
            rank = rank or op.get("rank")
        if not comp_key and "compress" in op_name:
            comp_key = (
                op.get("operator_key")
                or op.get("method")
                or op.get("compression_method")
                or op.get("compressor")
            )
            rank = rank or op.get("rank")
        if lora_rank is None and "lora" in op_name:
            lora_rank = op.get("rank") or op.get("r")

    if not attn_key and not comp_key and manifest.get("mechanism_key"):
        zoo_type = str(manifest.get("zoo_type") or "").lower()
        if "comp" in zoo_type:
            comp_key = manifest["mechanism_key"]
        else:
            attn_key = manifest["mechanism_key"]

    if pattern is None and attn_key:
        if any(".layers.3.self_attn.core." in k for k in state_dict):
            pattern = "all"
        elif any(".layers.2.self_attn.core." in k for k in state_dict) and not any(
            ".layers.1.self_attn.core." in k for k in state_dict
        ):
            pattern = "hybrid_1_1"
        else:
            pattern = "hybrid_3_1"

    return {
        "attention_type": attn_key,
        "compression_method": comp_key,
        "pattern": pattern or "hybrid_3_1",
        "rank": int(rank) if rank is not None else None,
        "lora_rank": int(lora_rank) if lora_rank is not None else None,
    }


def _apply_bert_attention_surgery(
    model: nn.Module,
    attn_key: str,
    rank: Optional[int],
    state_dict: dict[str, torch.Tensor],
) -> nn.Module:
    """Replay BERT attention surgery using BertAttentionAdapter or replace_attention."""
    uses_adapter = any(".attention.self.core." in k for k in state_dict)
    encoder_layers = getattr(getattr(getattr(model, "bert", model), "encoder", None), "layer", None)

    if uses_adapter and encoder_layers is not None:
        from neuralops.layers.bert_adapter import BertAttentionAdapter

        hidden_size = getattr(model.config, "hidden_size", 768)
        num_heads = getattr(model.config, "num_attention_heads", 12)
        for idx, layer in enumerate(encoder_layers):
            core = _instantiate_core_attention(
                attn_key=attn_key,
                hidden_size=hidden_size,
                num_heads=num_heads,
                text_cfg=model.config,
                rank=rank,
                layer_idx=idx,
            )
            orig_self = getattr(layer.attention, "self", None)
            layer.attention.self = BertAttentionAdapter(
                core_attention=core,
                embed_dim=hidden_size,
                num_heads=num_heads,
                inherit_weights_from=orig_self,
                bypass_internal_out_proj=True,
            )
        return model

    from neuralops.api import replace_attention

    if rank is not None:
        return replace_attention(model, attention=attn_key, rank=rank, copy_weights=False)
    return replace_attention(model, attention=attn_key, copy_weights=False)


def _attach_lora_adapters_from_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    default_lora_rank: Optional[int] = None,
) -> int:
    """Wrap modules whose checkpoint tensors include LoRA parameters (lora_A/lora_B)."""
    from neuralops.core.model_utils import replace_module

    lora_targets: dict[str, int] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A"):
            mod_path = key[: -len(".lora_A")]
            lora_targets[mod_path] = int(tensor.shape[0]) if tensor.ndim == 2 else int(default_lora_rank or 8)
        elif key.endswith(".lora_A.weight"):
            mod_path = key[: -len(".lora_A.weight")]
            lora_targets[mod_path] = int(tensor.shape[0]) if tensor.ndim == 2 else int(default_lora_rank or 8)

    wrapped_count = 0
    for mod_path, rank_val in lora_targets.items():
        try:
            existing = model.get_submodule(mod_path)
        except AttributeError:
            continue
        if hasattr(existing, "lora_A") and getattr(existing, "lora_A", None) is not None:
            continue
        if f"{mod_path}.lora_A.weight" in state_dict:
            from neuralops.trainer.methods.lora import LoRALinear

            if isinstance(existing, nn.Linear):
                replace_module(model, mod_path, LoRALinear(existing, r=rank_val))
                wrapped_count += 1
                continue
        replace_module(model, mod_path, _LoRALinearAdapter(existing, r=rank_val))
        wrapped_count += 1

    return wrapped_count


def _load_tokenizer_or_processor(
    repo_id: str,
    base_model_id: str,
    *,
    is_vlm: bool,
    trust_remote_code: bool,
    token: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
) -> Any:
    """Load AutoProcessor for VLMs or AutoTokenizer for text models with base fallback."""
    from transformers import AutoProcessor, AutoTokenizer

    load_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if token is not None:
        load_kw["token"] = token
    if revision is not None:
        load_kw["revision"] = revision
    if cache_dir is not None:
        load_kw["cache_dir"] = cache_dir

    loader_cls = AutoProcessor if is_vlm else AutoTokenizer
    for source in (repo_id, base_model_id):
        try:
            return loader_cls.from_pretrained(str(source), **load_kw)
        except Exception as exc:
            logger.debug("Failed loading %s from %s: %s", loader_cls.__name__, source, exc)
    if is_vlm:
        for source in (repo_id, base_model_id):
            try:
                return AutoTokenizer.from_pretrained(str(source), **load_kw)
            except Exception as exc:
                logger.debug("Fallback AutoTokenizer failed from %s: %s", source, exc)
    raise RuntimeError(f"Could not load tokenizer/processor from '{repo_id}' or base '{base_model_id}'.")


def load_surgeried_model(
    repo_id: str,
    device: str = "auto",
    torch_dtype: Any = None,
    task: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """Load any NeuralOps surgeried or compressed checkpoint from HF Hub or local directory.

    Reads ``config.json`` and optional ``surgical_manifest.json``, initializes the frozen
    base model, replays the recorded attention or compression surgery, downloads strictly
    ``attention_weights.safetensors`` (falling back to ``pytorch_model.bin`` only for
    legacy checkpoints), loads the delta weights with ``strict=False``, and returns
    ``(model.eval(), tokenizer_or_processor)``.
    """
    token = kwargs.pop("token", None) or kwargs.pop("hf_token", None) or os.environ.get("HF_TOKEN")
    revision = kwargs.pop("revision", None)
    cache_dir = kwargs.pop("cache_dir", None)
    base_model_override = kwargs.pop("base_model_id", None)
    num_labels_override = kwargs.pop("num_labels", None)

    cfg_path = _resolve_repo_file(
        repo_id, "config.json", optional=False, token=token, revision=revision, cache_dir=cache_dir
    )
    assert cfg_path is not None
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    manifest_path = _resolve_repo_file(
        repo_id,
        "surgical_manifest.json",
        optional=True,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path is not None else {}

    weights_filename = manifest.get("weights_file") or "attention_weights.safetensors"
    weights_path = _resolve_repo_file(
        repo_id,
        weights_filename,
        optional=True,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
    )
    if weights_path is None and weights_filename != "attention_weights.safetensors":
        weights_path = _resolve_repo_file(
            repo_id,
            "attention_weights.safetensors",
            optional=True,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
    if weights_path is None:
        weights_path = _resolve_repo_file(
            repo_id,
            "pytorch_model.bin",
            optional=True,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
        )
    if weights_path is None:
        raise FileNotFoundError(
            f"Neither '{weights_filename}', 'attention_weights.safetensors', nor 'pytorch_model.bin' found in '{repo_id}'."
        )

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
    else:
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    base_model_id = (
        normalize_base_model_id(base_model_override)
        if base_model_override
        else _infer_base_model_id(repo_id, cfg_dict, manifest)
    )
    surgery = _infer_surgery_spec(cfg_dict, manifest, state_dict)

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    resolved_dtype = _resolve_dtype(torch_dtype)
    if resolved_dtype is None and target_device.type == "cuda":
        if cfg_dict.get("weights_format", "").endswith("fp16") or cfg_dict.get("torch_dtype") == "float16":
            resolved_dtype = torch.float16

    model_cls, accepts_num_labels = _select_model_class(task, cfg_dict)
    base_load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    if token is not None:
        base_load_kwargs["token"] = token
    if cache_dir is not None:
        base_load_kwargs["cache_dir"] = cache_dir
    if resolved_dtype is not None:
        base_load_kwargs["torch_dtype"] = resolved_dtype
    if accepts_num_labels:
        resolved_labels = num_labels_override if num_labels_override is not None else cfg_dict.get("num_labels", 2)
        base_load_kwargs["num_labels"] = int(resolved_labels)

    logger.info("Loading base model '%s' (%s) for repo '%s'", base_model_id, model_cls.__name__, repo_id)
    model = model_cls.from_pretrained(base_model_id, **base_load_kwargs)

    model_type = (cfg_dict.get("model_type") or "").lower()
    attn_key = surgery["attention_type"]
    comp_key = surgery["compression_method"]

    if attn_key:
        if model_type == "bert" or hasattr(model, "bert"):
            model = _apply_bert_attention_surgery(
                model,
                attn_key=attn_key,
                rank=surgery["rank"],
                state_dict=state_dict,
            )
        else:
            model, _, _ = apply_decoder_attention_surgery(
                model,
                attn_key=attn_key,
                pattern=surgery["pattern"],
                rank=surgery["rank"],
                dtype=resolved_dtype,
            )
    elif comp_key:
        from neuralops.core.compression_pipeline import compress_model

        comp_rank = surgery["rank"] if surgery["rank"] is not None else 64
        comp_step: dict[str, Any] = {"method": comp_key, "rank": comp_rank}
        if model_type in ("qwen2", "llama", "mistral") or comp_key == "aa_svd":
            comp_step.update({"force_rank": True, "use_strs": False})
        try:
            model = compress_model(model, {"compressors": [comp_step]})
        except TypeError:
            model = compress_model(model, {"compressors": [{"method": comp_key, "rank": comp_rank}]})

        _attach_lora_adapters_from_state_dict(
            model,
            state_dict,
            default_lora_rank=surgery["lora_rank"],
        )
    elif manifest.get("submodules"):
        from neuralops.models.surgical_io import load_surgical_checkpoint

        if Path(repo_id).is_dir():
            model, _ = load_surgical_checkpoint(model, repo_id, map_location="cpu", strict=False)
            state_dict = {}

    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            logger.debug("Unexpected keys while loading %s: %s", repo_id, unexpected[:10])
        if missing:
            logger.debug("Base/unmodified keys retained from %s (%d keys)", base_model_id, len(missing))

    if resolved_dtype is not None:
        model = model.to(device=target_device, dtype=resolved_dtype)
    else:
        model = model.to(device=target_device)
    model.eval()

    is_vlm = model_type == "llava" or task in ("vision-language", "vlm")
    tokenizer_or_processor = _load_tokenizer_or_processor(
        repo_id=repo_id,
        base_model_id=base_model_id,
        is_vlm=is_vlm,
        trust_remote_code=trust_remote_code,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
    )
    return model, tokenizer_or_processor


__all__ = [
    "normalize_base_model_id",
    "apply_decoder_attention_surgery",
    "load_surgeried_model",
]
