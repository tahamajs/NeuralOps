#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.compression."""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from typing import Any

import neuralops.compression as _mod

COMPRESSOR_REGISTRY = _mod.COMPRESSOR_REGISTRY
get_compressor = _mod.get_compressor
list_compressors = _mod.list_compressors
BaseCompressor = _mod.BaseCompressor
compress_model = _mod.compress_model

_SUBPACKAGES = _mod._SUBPACKAGES


class _CompressionEngineRedirectFinder(MetaPathFinder):
    """Redirects imports from neuralops.compression_engine.<name> to neuralops.compression.<name> or subpackages."""
    prefix = "neuralops.compression_engine."

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if not fullname.startswith(self.prefix):
            return None
        subname = fullname[len(self.prefix):]
        candidates = [
            f"neuralops.compression.{subname}",
            f"neuralops.compression.svd.{subname}",
            f"neuralops.compression.quantization.{subname}",
            f"neuralops.compression.pruning.{subname}",
            f"neuralops.compression.low_rank.{subname}",
        ]
        for cand in candidates:
            try:
                spec = importlib.util.find_spec(cand)
                if spec is not None:
                    return importlib.util.spec_from_loader(fullname, _AliasLoader(cand))
            except Exception:
                continue
        return None


class _AliasLoader(Loader):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def create_module(self, spec: Any) -> Any:
        return importlib.import_module(self.target_name)

    def exec_module(self, module: Any) -> None:
        pass


if not any(isinstance(finder, _CompressionEngineRedirectFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _CompressionEngineRedirectFinder())


def __getattr__(name: str) -> Any:
    return getattr(_mod, name)


def __dir__() -> list[str]:
    return list(_mod.__dir__())


__all__ = list(_mod.__all__)
