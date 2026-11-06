#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Static catalog inspection of registered attention and compressor operators."""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def _extract_family(cls: type) -> str:
    """Extract family name from module path: e.g. neuralops.layers.linear.cosformer -> linear."""
    module_parts = getattr(cls, "__module__", "").split(".")
    if len(module_parts) >= 3 and module_parts[1] == "layers":
        return module_parts[2]
    return "unknown"


def _extract_paradigm(cls: type) -> str:
    """Extract compression paradigm from module path: e.g. neuralops.compression.svd.direct_svd -> svd."""
    module_parts = getattr(cls, "__module__", "").split(".")
    if len(module_parts) >= 3 and module_parts[1] == "compression":
        return module_parts[2]
    return "unknown"


def list_attentions(family: str | None = None) -> list[dict[str, Any]]:
    """List registered attention mechanisms, optionally filtered by family."""
    results: list[dict[str, Any]] = []
    for name, cls in ATTENTION_REGISTRY.items():
        fam = _extract_family(cls)
        if family is not None and fam != family:
            continue
        doc = (cls.__doc__ or "").strip()
        results.append(
            {
                "name": name,
                "family": fam,
                "module": getattr(cls, "__module__", ""),
                "doc": doc,
            }
        )
    results.sort(key=lambda item: item["name"])
    return results


def list_compressors(paradigm: str | None = None) -> list[dict[str, Any]]:
    """List registered compressors, optionally filtered by compression paradigm."""
    results: list[dict[str, Any]] = []
    for name, cls in COMPRESSOR_REGISTRY.items():
        par = _extract_paradigm(cls)
        if paradigm is not None and par != paradigm:
            continue
        doc = (cls.__doc__ or "").strip()
        results.append(
            {
                "name": name,
                "paradigm": par,
                "module": getattr(cls, "__module__", ""),
                "doc": doc,
            }
        )
    results.sort(key=lambda item: item["name"])
    return results


def get_attention_metadata(name: str) -> dict[str, Any] | None:
    """Retrieve metadata for a specific attention mechanism."""
    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        return None
    return {
        "name": name,
        "family": _extract_family(cls),
        "module": getattr(cls, "__module__", ""),
        "doc": (cls.__doc__ or "").strip(),
        "class_name": getattr(cls, "__name__", str(cls)),
    }


def get_compressor_metadata(name: str) -> dict[str, Any] | None:
    """Retrieve metadata for a specific compressor."""
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        return None
    return {
        "name": name,
        "paradigm": _extract_paradigm(cls),
        "module": getattr(cls, "__module__", ""),
        "doc": (cls.__doc__ or "").strip(),
        "class_name": getattr(cls, "__name__", str(cls)),
    }


def list_families() -> list[str]:
    """Return sorted unique list of all attention families."""
    families = {_extract_family(cls) for cls in ATTENTION_REGISTRY.values()}
    return sorted(list(families))


def list_paradigms() -> list[str]:
    """Return sorted unique list of all compressor paradigms."""
    paradigms = {_extract_paradigm(cls) for cls in COMPRESSOR_REGISTRY.values()}
    return sorted(list(paradigms))
