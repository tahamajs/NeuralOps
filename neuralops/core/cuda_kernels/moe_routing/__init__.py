#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mixture of Experts Routing Hardware Operators.
"""

from neuralops.core.cuda_kernels.moe_routing.moe_autograd import (
    moe_dispatch_gather,
    moe_dispatch,
    moe_gather,
    moe_routing,
    moe_routing_ref,
    moe_routing_pytorch_ref,
    moe_dispatch_gather_ref,
    moe_dispatch_gather_pytorch_ref,
    moe_route_pytorch_ref,
    _moe_dispatch_pytorch_ref,
    _moe_gather_pytorch_ref,
)

__all__ = [
    "moe_dispatch_gather",
    "moe_dispatch",
    "moe_gather",
    "moe_routing",
    "moe_routing_ref",
    "moe_routing_pytorch_ref",
    "moe_dispatch_gather_ref",
    "moe_dispatch_gather_pytorch_ref",
    "moe_route_pytorch_ref",
    "_moe_dispatch_pytorch_ref",
    "_moe_gather_pytorch_ref",
]
