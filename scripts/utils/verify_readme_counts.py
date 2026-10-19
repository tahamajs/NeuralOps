#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verify the release registry, kernel taxonomy, and notebook invariants."""

from __future__ import annotations

from pathlib import Path

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.core.triton_kernels import (
    TRITON_CORE_MODULES,
    TRITON_EXTENDED_KERNELS,
    TRITON_ROOT_MODULES,
    TRITON_STANDALONE_KERNELS,
)


def main() -> int:
    """Return a nonzero status when a canonical invariant is violated."""
    root = Path(__file__).resolve().parents[2]
    attention_count = len(ATTENTION_REGISTRY._dict)
    compressor_count = len(COMPRESSOR_REGISTRY._dict)
    triton_count = sum(
        len(group)
        for group in (
            TRITON_STANDALONE_KERNELS,
            TRITON_CORE_MODULES,
            TRITON_ROOT_MODULES,
            TRITON_EXTENDED_KERNELS,
        )
    )
    notebook_count = len(list((root / "notebooks").glob("*.ipynb")))
    observed = {
        "attention_registry": attention_count,
        "compressor_registry": compressor_count,
        "triton_operator_taxonomy": triton_count,
        "notebooks": notebook_count,
    }
    expected = {
        "attention_registry": 131,
        "compressor_registry": 72,
        "triton_operator_taxonomy": 62,
        "notebooks": 27,
    }
    print(f"Observed invariants: {observed}")
    if observed != expected:
        print(f"Expected invariants: {expected}")
        return 1
    print("All canonical README invariants verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
