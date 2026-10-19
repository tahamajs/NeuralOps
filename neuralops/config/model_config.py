#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Architecture Configuration

Specification of backbone transformer model types, layer counts, dimensions, and hidden states.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    attention_type: str = "full"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: str = "./cache"
    fix_layernorm_naming: bool = True
    variants: list[str] = field(default_factory=list)
