#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shared PEP 562 Lazy Loading Factory

Helper function to create __getattr__ and __dir__ for subpackages with dictionary mappings.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple


def create_lazy_getattr(
    package_name: str,
    symbol_map: Dict[str, Tuple[str, str]],
    extra_globals: Dict[str, Any] | None = None,
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Build a standard __getattr__ and __dir__ pair for a package."""
    extra = extra_globals or {}

    def __getattr__(name: str) -> Any:
        if name in extra:
            return extra[name]
        if name in symbol_map:
            mod_suffix, attr_name = symbol_map[name]
            module = importlib.import_module(f"{package_name}.{mod_suffix}")
            val = getattr(module, attr_name)
            extra[name] = val
            return val
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    def __dir__() -> List[str]:
        return sorted(list(symbol_map.keys()) + list(extra.keys()))

    return __getattr__, __dir__
