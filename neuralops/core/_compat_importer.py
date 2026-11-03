#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transparent In-Memory Import Interceptor for NeuralOps Legacy Subpackages.
Ensures identical module identity (A is B) without maintaining duplicate files.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from importlib.abc import Loader, MetaPathFinder
from types import ModuleType
from typing import Any, Optional, Sequence

logger = logging.getLogger("NeuralOps.CompatImporter")


class ExactModuleAliasLoader(Loader):
    """Loader that delegates module resolution directly to the canonical module in sys.modules."""

    def __init__(self, canonical_name: str, is_pkg: bool = False):
        self.canonical_name = canonical_name
        self.is_pkg = is_pkg

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        if self.canonical_name not in sys.modules:
            importlib.import_module(self.canonical_name)
        return sys.modules[self.canonical_name]

    def exec_module(self, module: ModuleType) -> None:
        pass


class NeuralOpsCompatibilityRedirectFinder(MetaPathFinder):
    """Intercepts legacy subpackage requests and resolves them to canonical packages with identity preservation."""

    _EXACT_MAP = {
        "neuralops.attentions": ("neuralops.layers", True),
        "neuralops.compression_engine": ("neuralops.compression", True),
        "neuralops.training": ("neuralops.trainer", True),
        "neuralops.serving": ("neuralops.export.serving", True),
        "neuralops.backward_compat": ("neuralops.core", True),
    }

    _PREFIX_MAP = {
        "neuralops.attentions.": "neuralops.layers.",
        "neuralops.compression_engine.": "neuralops.compression.",
        "neuralops.training.": "neuralops.trainer.",
        "neuralops.serving.": "neuralops.export.serving.",
        "neuralops.backward_compat.attentions.": "neuralops.layers.",
        "neuralops.backward_compat.compression_engine.": "neuralops.compression.",
    }

    def _resolve_canonical(self, canonical_fullname: str) -> Optional[tuple[str, bool]]:
        # Try direct
        try:
            c_spec = importlib.util.find_spec(canonical_fullname)
            if c_spec is not None:
                is_pkg = c_spec.submodule_search_locations is not None
                return (canonical_fullname, is_pkg)
        except Exception:
            pass

        # If flat submodule lookup failed in layers or compression, try searching subdirectories
        parts = canonical_fullname.split(".")
        if len(parts) == 3 and parts[0] == "neuralops":
            pkg, mod = parts[1], parts[2]
            subpkgs = []
            if pkg == "layers":
                subpkgs = ['linear', 'svd', 'sparse', 'gated', 'dense', 'ssm', 'hybrid', 'modern', 'adaptive', 'fast_kernels', 'experimental']
            elif pkg == "compression":
                subpkgs = ['svd', 'quantization', 'pruning', 'low_rank', 'tensor_decomp', 'kv_cache', 'experimental']
            
            for sub in subpkgs:
                candidate = f"neuralops.{pkg}.{sub}.{mod}"
                try:
                    c_spec = importlib.util.find_spec(candidate)
                    if c_spec is not None:
                        is_pkg = c_spec.submodule_search_locations is not None
                        return (candidate, is_pkg)
                except Exception:
                    continue
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[Any]:
        # 1. Exact package root matching
        if fullname in self._EXACT_MAP:
            target_name, is_pkg = self._EXACT_MAP[fullname]
            loader = ExactModuleAliasLoader(target_name, is_pkg=is_pkg)
            return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)

        # 2. Submodule prefix matching
        for old_prefix, new_prefix in self._PREFIX_MAP.items():
            if fullname.startswith(old_prefix):
                subpath = fullname[len(old_prefix):]
                canonical_fullname = f"{new_prefix}{subpath}"
                resolved = self._resolve_canonical(canonical_fullname)
                if resolved is not None:
                    canon, is_pkg = resolved
                    loader = ExactModuleAliasLoader(canon, is_pkg=is_pkg)
                    return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)
        return None


def install_compatibility_interceptor() -> None:
    """Installs the compatibility interceptor at the front of sys.meta_path idempotently."""
    if not any(isinstance(finder, NeuralOpsCompatibilityRedirectFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, NeuralOpsCompatibilityRedirectFinder())
        logger.debug("NeuralOpsCompatibilityRedirectFinder active in sys.meta_path.")
