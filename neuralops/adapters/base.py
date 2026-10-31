#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Adapter Abstract Base Class.

Implements introspective model structure discovery and wildcard target resolution.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


class ArchitectureAdapter(ABC):
    """Abstract Base Class for model family architecture adapters."""

    family: str
    supported_versions: Dict[str, str]
    tested_model_ids: List[str]

    @abstractmethod
    def can_handle(self, model: nn.Module) -> bool:
        """Return True if this adapter can introspect and resolve modules for model."""
        raise NotImplementedError

    @abstractmethod
    def detect_family(self, model: nn.Module) -> Optional[str]:
        """Detect model family name or return None if unknown."""
        raise NotImplementedError

    @abstractmethod
    def inspect(self, model: nn.Module) -> Dict[str, Any]:
        """Inspect model architecture dimensions, layer count, and module map."""
        raise NotImplementedError

    def resolve_target(self, model: nn.Module, target_spec: str) -> List[Tuple[str, nn.Module]]:
        """Resolve dotted or wildcard path specification into list of (path, module) pairs.

        Example:
            'distilbert.transformer.layer.*.attention.q_lin' -> [('distilbert...', module), ...]
        """
        matched: List[Tuple[str, nn.Module]] = []
        for name, mod in model.named_modules():
            if fnmatch.fnmatch(name, target_spec) or fnmatch.fnmatch(f"*.{name}", target_spec):
                matched.append((name, mod))
        return matched

    @abstractmethod
    def get_capabilities(self) -> Dict[str, str]:
        """Return specific mapping of supported operations to empirical support level."""
        raise NotImplementedError

    def declared_capabilities(self) -> Dict[str, str]:
        """Return mapping of declared operation capabilities."""
        return self.get_capabilities()

    def check_precondition(self, model: nn.Module, precondition_name: str) -> Tuple[bool, str]:
        """Verify whether model satisfies the named precondition.

        Returns:
            Tuple of (satisfied: bool, reason_if_failed: str).
        """
        config = getattr(model, "config", None)
        if precondition_name == "has_standard_attention":
            has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in model.named_modules())
            return (has_attn, "" if has_attn else "Model has no recognizable attention modules")

        if precondition_name == "model.has_attention":
            has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in model.named_modules())
            return (has_attn, "" if has_attn else "Model has no attention blocks")

        if precondition_name == "uncompressed_linear_weights_exist":
            has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
            return (has_linear, "" if has_linear else "No uncompressed nn.Linear layers present")

        if precondition_name == "has_standard_mlp_blocks":
            has_mlp = any("mlp" in n.lower() or "feed_forward" in n.lower() or "ffn" in n.lower() for n, _ in model.named_modules())
            return (has_mlp, "" if has_mlp else "No standard MLP blocks detected")

        if precondition_name == "num_layers > drop_count":
            num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
            return (num_layers > 1, "" if num_layers > 1 else f"Insufficient layers ({num_layers}) for dropping")

        if "hidden_size % head_dim == 0" in precondition_name:
            h_size = getattr(config, "hidden_size", getattr(config, "d_model", 768))
            # Default check for standard head dim 64
            ok = (h_size % 64 == 0)
            return (ok, "" if ok else f"hidden_size {h_size} not divisible by head_dim 64")

        # Fail-closed for unrecognized precondition names
        return (False, f"precondition_checker_unavailable:{precondition_name}")
