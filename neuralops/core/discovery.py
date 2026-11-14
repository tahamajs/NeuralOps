#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Core Module Auto-Discovery & Capability Inspection.

Exposes canonical discovery contracts, capability introspection, and metadata reflection.
"""

from __future__ import annotations

from neuralops.benchmarks.discovery import (
    ModuleMetadata,
    discover_all_modules,
    discover_modules,
    inspect_module_capabilities,
)

__all__ = [
    "ModuleMetadata",
    "inspect_module_capabilities",
    "discover_all_modules",
    "discover_modules",
]
