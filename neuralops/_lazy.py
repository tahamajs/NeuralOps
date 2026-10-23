#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Lazy Module Infrastructure

Hugging Face Transformers-style _LazyModule that defers module loading until symbol access.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional, Set


class _LazyModule(ModuleType):
    """Module that only imports child modules and symbols when accessed."""

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: Dict[str, List[str]],
        module_spec: Any = None,
        extra_objects: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name)
        self._modules: Set[str] = set(import_structure.keys())
        self._class_to_module: Dict[str, str] = {}
        for mod, fqns in import_structure.items():
            for fqn in fqns:
                self._class_to_module[fqn] = mod
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects: Dict[str, Any] = extra_objects or {}
        self.__all__ = (
            list(import_structure.keys())
            + [val for vals in import_structure.values() for val in vals]
            + list(self._objects.keys())
        )

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            available = ", ".join(sorted(self.__all__[:15]))
            more = f" ... and {len(self.__all__) - 15} more" if len(self.__all__) > 15 else ""
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}. Available: {available}{more}"
            )
        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error:\n{e}"
            ) from e

    def __dir__(self) -> List[str]:
        result = list(super().__dir__())
        for k in self.__all__:
            if k not in result:
                result.append(k)
        return sorted(result)
