#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LLaMA, Mistral, and Qwen2 Architecture Adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import register_adapter


class LlamaAdapter(ArchitectureAdapter):
    family = "llama"
    supported_versions = {"transformers": ">=4.40.0"}
    tested_model_ids = ["meta-llama/Llama-3-8B", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.1"]

    def can_handle(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        if config is not None:
            m_type = getattr(config, "model_type", "").lower()
            if m_type in ("llama", "mistral", "qwen2", "qwen"):
                return True
        c_name = model.__class__.__name__.lower()
        return "llama" in c_name or "mistral" in c_name or "qwen" in c_name

    def detect_family(self, model: nn.Module) -> Optional[str]:
        return self.family if self.can_handle(model) else None

    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_heads = getattr(config, "num_attention_heads", 32)

        modules_map = {n: m.__class__.__name__ for n, m in model.named_modules() if len(list(m.children())) == 0}
        return {
            "family": self.family,
            "architecture": model.__class__.__name__,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "modules": modules_map,
        }

    def get_capabilities(self) -> Dict[str, str]:
        return {
            "attention.replace[target=chunk_gla]": "tested_on_llama_3_8b_only",
            "weights.compress[method=aasvd]": "tested",
            "architecture.reduce_mlp": "tested",
            "architecture.drop_layer": "tested",
            "weights.quantize[method=int4_awq]": "requires_cuda",
        }


register_adapter(LlamaAdapter())
