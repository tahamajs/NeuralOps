#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Abstraction Protocols and Surgical State Contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn


class ModelHandle(ABC):
    """Abstract interface for inspecting and mutating transformer models."""

    @property
    @abstractmethod
    def base_model(self) -> nn.Module:
        """Returns the underlying PyTorch nn.Module."""

    @abstractmethod
    def find_attention_layers(self) -> Dict[str, nn.Module]:
        """Finds all attention modules mapped by their dot-separated submodule path."""

    @abstractmethod
    def find_mlp_layers(self) -> Dict[str, nn.Module]:
        """Finds all MLP modules mapped by their dot-separated submodule path."""

    @abstractmethod
    def capture_manifest(self) -> Dict[str, Any]:
        """Captures a complete surgical manifest describing layer topologies and modifications."""


class SurgicalModel(ABC):
    """Protocol for models that have undergone surgical attention replacement or compression."""

    @abstractmethod
    def get_surgical_manifest(self) -> Dict[str, Any]:
        """Returns the serialized surgical manifest associated with this model."""

    @abstractmethod
    def get_attention_family(self, layer_idx: int) -> str:
        """Returns the canonical attention key implemented at layer_idx."""
