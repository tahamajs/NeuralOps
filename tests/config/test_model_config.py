#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.config.model_config."""

import pytest

from neuralops.config import ModelConfig


class TestModelConfig:
    """Test ModelConfig dataclass fields and defaults."""

    def test_creation(self):
        cfg = ModelConfig()
        assert cfg is not None

    def test_custom_name(self):
        cfg = ModelConfig(name="bert-base-uncased")
        assert cfg.name == "bert-base-uncased"

    def test_num_labels(self):
        cfg = ModelConfig(num_labels=3)
        assert cfg.num_labels == 3

    def test_variants_field(self):
        cfg = ModelConfig()
        assert hasattr(cfg, "variants")
        assert isinstance(cfg.variants, list)
