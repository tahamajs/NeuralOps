#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Core Configuration Parser and Standard Configuration Objects.

Lightweight YAML configuration loader and canonical Config / DummyConfig
containers for core execution settings, model definitions, and tests.
"""

from __future__ import annotations

from typing import Any

import yaml


class DummyConfig:
    """Canonical lightweight dummy configuration for testing and fallback initialization."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 512,
        vocab_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.d_model = int(hidden_size)
        self.dim = int(hidden_size)
        heads = int(num_attention_heads if num_attention_heads is not None else num_heads)
        self.num_attention_heads = heads
        self.num_heads = heads
        self.n_heads = heads
        self.num_key_value_heads = int(
            num_key_value_heads if num_key_value_heads is not None else heads
        )
        self.head_dim = max(1, self.hidden_size // max(1, heads))
        self.intermediate_size = int(
            intermediate_size if intermediate_size is not None else self.hidden_size * 4
        )
        self.max_position_embeddings = int(max_position_embeddings)
        self.vocab_size = int(vocab_size)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __getattr__(self, item: str) -> Any:
        return None


class Config:
    """Canonical root configuration container uniting subsystem configurations."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __getattr__(self, item: str) -> Any:
        return None


def load_config(path: str) -> dict[str, Any]:
    """Load config from disk or config."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str) -> None:
    """Persist config to disk."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


__all__ = ["load_config", "save_config", "Config", "DummyConfig"]
