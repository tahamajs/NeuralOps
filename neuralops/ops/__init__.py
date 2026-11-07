#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Operator Tier (Tier 1)

Manages native CUDA C++, Triton JIT, and PyTorch reference kernel execution.
"""

from __future__ import annotations

import importlib
from typing import Any

_import_structure = {
    "cuda": ["list_cuda_kernels", "CUDA_KERNEL_CATALOG"],
    "triton": ["triton_dispatch"],
    "pytorch": [
        "rmsnorm_pytorch_ref",
        "moe_routing_pytorch_ref",
        "moe_dispatch_gather_pytorch_ref",
        "fused_rotary_emb_pytorch_ref",
        "fused_cross_entropy_pytorch_ref",
        "fused_kd_loss_pytorch_ref",
        "chunked_delta_gate_scan_pytorch_ref",
        "gla_scan_pytorch_ref",
        "selective_scan_pytorch_ref",
    ],
}

__all__ = ["cuda", "triton", "pytorch"]


def __getattr__(name: str) -> Any:
    if name in _import_structure:
        mod = importlib.import_module(f".{name}", __name__)
        return mod
    for mod_name, symbols in _import_structure.items():
        if name in symbols:
            mod = importlib.import_module(f".{mod_name}", __name__)
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
