#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke tests for the eager and lazy NeuralOps public API surface."""

from __future__ import annotations

import importlib

import neuralops


def test_all_public_exports_resolve() -> None:
    """Every declared public export must resolve without an import error."""
    failures: list[str] = []
    for name in neuralops.__all__:
        try:
            value = getattr(neuralops, name)
            if value is None:
                failures.append(f"{name}: resolved to None")
        except Exception as exc:  # pragma: no cover - failure detail is asserted below
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Public API export failures: " + "; ".join(failures)


def test_lazy_export_targets_resolve() -> None:
    """All lazy export maps must point to importable attributes."""
    failures: list[str] = []
    maps = [getattr(neuralops, "_LAZY_EXPORTS"), getattr(neuralops, "_API_EXPORTS")]
    for export_map in maps:
        for name, target in export_map.items():
            module_name, attribute_name = target if isinstance(target, tuple) else (target, name)
            try:
                module = importlib.import_module(module_name)
                value = getattr(module, attribute_name)
                if value is None:
                    failures.append(f"{name}: {module_name}.{attribute_name} is None")
            except Exception as exc:  # pragma: no cover - failure detail is asserted below
                failures.append(f"{name}: {module_name}.{attribute_name}: {type(exc).__name__}: {exc}")
    assert not failures, "Lazy export failures: " + "; ".join(failures)


def test_core_api_exports_are_callable() -> None:
    """The core user-facing factory and surgery functions remain callable."""
    names = (
        "replace_attention",
        "compress",
        "create_attention",
        "create_compressor",
        "list_attentions",
        "list_compressors",
        "list_triton_kernels",
        "list_cuda_kernels",
        "heal_replaced_model",
        "calibrate_attention_subspace",
        "profile_roofline",
        "probe_optimal_batch_size",
    )
    assert all(callable(getattr(neuralops, name)) for name in names)
