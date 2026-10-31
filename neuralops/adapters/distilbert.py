#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
DistilBERT and BERT Architecture Adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import register_adapter


class DistilBertAdapter(ArchitectureAdapter):
    family = "bert"
    supported_versions = {"transformers": ">=4.30.0"}
    tested_model_ids = ["distilbert-base-uncased", "distilbert-base-cased"]

    def can_handle(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        if config is not None:
            m_type = getattr(config, "model_type", "").lower()
            if m_type in ("distilbert", "bert", "roberta"):
                return True
        c_name = model.__class__.__name__.lower()
        return "distilbert" in c_name or "bert" in c_name

    def detect_family(self, model: nn.Module) -> Optional[str]:
        return self.family if self.can_handle(model) else None

    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", 6))
        num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))

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
            "weights.compress[method=aasvd]": "tested",
            "weights.compress[method=svd]": "tested",
            "architecture.drop_layer": "tested",
            "architecture.reduce_mlp": "tested",
            "attention.replace[target=chunk_gla]": "experimental",
            "weights.quantize[method=fp8]": "unsupported_on_cpu",
        }


# Register singleton instance
register_adapter(DistilBertAdapter())
