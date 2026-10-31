#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPT-2 Architecture Adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import register_adapter


class Gpt2Adapter(ArchitectureAdapter):
    family = "gpt2"
    supported_versions = {"transformers": ">=4.30.0"}
    tested_model_ids = ["gpt2", "distilgpt2", "gpt2-medium"]

    def can_handle(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        if config is not None:
            m_type = getattr(config, "model_type", "").lower()
            if m_type == "gpt2":
                return True
        c_name = model.__class__.__name__.lower()
        return "gpt2" in c_name

    def detect_family(self, model: nn.Module) -> Optional[str]:
        return self.family if self.can_handle(model) else None

    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "n_embd", 768)
        num_layers = getattr(config, "n_layer", 12)
        num_heads = getattr(config, "n_head", 12)

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
            "attention.replace[target=chunk_gla]": "tested",
            "weights.compress[method=aasvd]": "tested",
            "architecture.reduce_mlp": "tested",
            "architecture.drop_layer": "tested",
        }


register_adapter(Gpt2Adapter())
