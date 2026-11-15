#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hugging Face Transformers Architecture Adapter and Bridge.

Provides dynamic introspection, target resolution, parameter extraction, and
integration bridge for Hugging Face PreTrainedModel architectures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import register_adapter
from neuralops.core.adapters import UniversalAdapter
from neuralops.utils.lora import DEFAULT_TARGET_MODULES, find_all_linear_submodule_names

logger = logging.getLogger("neuralops.adapters.hf_transformers")


class HFTransformersAdapter(ArchitectureAdapter):
    """Universal architecture adapter and inspection bridge for Hugging Face PreTrainedModel instances."""

    family = "hf_transformers"
    supported_versions = {"transformers": ">=4.30.0"}
    tested_model_ids = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "gpt2",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "Qwen/Qwen2.5-7B",
        "t5-small",
    ]

    def can_handle(self, model: nn.Module) -> bool:
        """Check if model is a Hugging Face PreTrainedModel or contains standard HF attributes.

        Args:
            model: PyTorch module.

        Returns:
            True if model conforms to HF architecture patterns, False otherwise.
        """
        # 1. Check for PreTrainedModel base or attributes
        if hasattr(model, "config") and hasattr(model, "base_model_prefix"):
            return True

        # 2. Check for PretrainedConfig object
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "model_type"):
            return True

        # 3. Check for standard HF method signatures
        if hasattr(model, "get_input_embeddings") or hasattr(model, "get_output_embeddings"):
            return True

        # 4. Check class name heuristics
        cls_name = model.__class__.__name__.lower()
        if any(
            arch in cls_name
            for arch in (
                "forcausallm",
                "formaskedlm",
                "forsequenceclassification",
                "fortokenclassification",
                "forquestionanswering",
                "pretrainedmodel",
                "transformer",
                "gpt2",
                "llama",
                "bert",
                "t5",
            )
        ):
            return True

        return False

    def detect_family(self, model: nn.Module) -> Optional[str]:
        """Detect the specific Hugging Face model family or return 'hf_transformers'.

        Args:
            model: PyTorch module.

        Returns:
            Family string identifier or None if model cannot be handled.
        """
        if not self.can_handle(model):
            return None

        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "model_type"):
            return str(cfg.model_type).lower()

        cls_name = model.__class__.__name__.lower()
        for arch in ("llama", "mistral", "qwen", "gemma", "gpt2", "bert", "distilbert", "roberta", "t5", "falcon"):
            if arch in cls_name:
                return arch

        return self.family

    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        """Inspect model architecture dimensions, layer count, projections, and attention topology.

        Args:
            model: PyTorch module.

        Returns:
            Dictionary containing model structural summary.
        """
        cfg = getattr(model, "config", None)
        detected_family = self.detect_family(model) or "unknown"

        hidden_size = getattr(
            cfg,
            "hidden_size",
            getattr(cfg, "n_embd", getattr(cfg, "d_model", getattr(cfg, "dim", None))),
        )
        num_layers = getattr(
            cfg,
            "num_hidden_layers",
            getattr(cfg, "n_layer", getattr(cfg, "n_layers", getattr(cfg, "num_layers", None))),
        )
        num_heads = getattr(
            cfg,
            "num_attention_heads",
            getattr(cfg, "n_head", getattr(cfg, "n_heads", getattr(cfg, "num_heads", None))),
        )
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        vocab_size = getattr(cfg, "vocab_size", None)

        # Dynamic module catalog
        linear_leaves = find_all_linear_submodule_names(model)
        modules_map = {name: mod.__class__.__name__ for name, mod in model.named_modules() if not list(mod.children())}

        return {
            "family": detected_family,
            "architecture": model.__class__.__name__,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "vocab_size": vocab_size,
            "linear_leaf_modules": sorted(list(linear_leaves)),
            "modules_count": len(modules_map),
            "is_decoder": getattr(cfg, "is_decoder", False),
            "is_encoder_decoder": getattr(cfg, "is_encoder_decoder", False),
        }

    def get_capabilities(self) -> Dict[str, str]:
        """Return support matrix for architecture operations.

        Returns:
            Dictionary mapping operation signatures to support levels ('tested', 'experimental', etc.).
        """
        return {
            "peft.lora_adaptation": "tested",
            "peft.roundtrip_fidelity": "tested",
            "weights.compress[method=svd]": "tested",
            "weights.compress[method=lora]": "tested",
            "attention.inspect_projections": "tested",
            "architecture.hook_injection": "tested",
        }

    def extract_projection_weights(
        self,
        attention_module: nn.Module,
        proj: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract projection weight and optional bias using UniversalAdapter.

        Args:
            attention_module: Attention module (e.g. LlamaAttention, BertSelfAttention, GPT2Attention).
            proj: Projection key ('q', 'k', 'v', or 'o').

        Returns:
            Tuple of (weight_tensor, optional_bias_tensor).

        Raises:
            ValueError: If projection cannot be extracted.
        """
        return UniversalAdapter.get_weight_bias(attention_module, proj)

    def get_suggested_lora_targets(self, model: nn.Module) -> List[str]:
        """Discover and return recommended LoRA target linear module names.

        Args:
            model: PyTorch module.

        Returns:
            List of unique leaf submodule names suitable for LoRA adaptation.
        """
        all_linears = find_all_linear_submodule_names(model)
        common_defaults = set(DEFAULT_TARGET_MODULES)

        # Intersect with common targets if found, otherwise return all discovered linears
        matched = all_linears.intersection(common_defaults)
        if matched:
            return sorted(list(matched))
        return sorted(list(all_linears))


# Register singleton adapter into global registry
register_adapter(HFTransformersAdapter())
