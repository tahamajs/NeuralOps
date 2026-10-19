# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Configuration Extraction Utilities

Helper functions extracting architecture parameters, head dimensions, and layer counts from configurations.
"""

from __future__ import annotations

from typing import Any

__all__ = ["_cfg_get"]


def _cfg_get(config_or_key: Any, key_or_default: Any = None, default: Any = None) -> Any:
    """Safely extract a configuration attribute.
    
    Supports both signatures:
      1. Standard: _cfg_get(config, key, default=None)
      2. Key-first / single-arg: _cfg_get(key, default=None) -> returns default
    """
    if isinstance(config_or_key, str):
        return key_or_default

    config = config_or_key
    key = str(key_or_default) if key_or_default is not None else ""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
