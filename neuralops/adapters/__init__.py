#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Adapters Subpackage.
"""

from neuralops.adapters.base import ArchitectureAdapter

# Eagerly register concrete adapters
from neuralops.adapters.distilbert import DistilBertAdapter
from neuralops.adapters.errors import AdapterError
from neuralops.adapters.generic import GenericAdapter
from neuralops.adapters.gpt2 import Gpt2Adapter
from neuralops.adapters.llama import LlamaAdapter
from neuralops.adapters.registry import find_adapter, list_adapters, register_adapter

__all__ = [
    "AdapterError",
    "ArchitectureAdapter",
    "DistilBertAdapter",
    "GenericAdapter",
    "Gpt2Adapter",
    "LlamaAdapter",
    "find_adapter",
    "list_adapters",
    "register_adapter",
]
