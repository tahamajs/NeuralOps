#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Generic Linear Architecture Adapter (Fallback for standalone dense networks only).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import register_adapter


class GenericAdapter(ArchitectureAdapter):
    family = "generic"
    supported_versions = {"torch": ">=2.0.0"}
    tested_model_ids = ["generic-linear-mlp"]

    def can_handle(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        if config is not None:
            m_type = getattr(config, "model_type", "").lower()
            # If the model explicitly declares a transformer or non-linear architecture, fail closed
            if m_type in ("mamba", "rwkv", "cnn", "unknown_xyz") or m_type.startswith("unknown"):
                return False
            if m_type in ("llama", "mistral", "qwen", "qwen2", "bert", "distilbert", "gpt2"):
                return False

        # Generic adapter strictly requires nn.Linear AND no attention blocks
        has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
        has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in model.named_modules())
        return has_linear and not has_attn

    def detect_family(self, model: nn.Module) -> Optional[str]:
        return self.family if self.can_handle(model) else None

    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        modules_map = {n: m.__class__.__name__ for n, m in model.named_modules() if len(list(m.children())) == 0}
        return {
            "family": self.family,
            "architecture": model.__class__.__name__,
            "modules": modules_map,
        }

    def get_capabilities(self) -> Dict[str, str]:
        return {
            "weights.compress[method=svd]": "all_linear",
            "weights.quantize[method=fp8]": "all_linear",
        }


register_adapter(GenericAdapter())
