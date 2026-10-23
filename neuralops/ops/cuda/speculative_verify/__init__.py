#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Speculative Verification Hardware Operators.
"""

from neuralops.ops.cuda.speculative_verify.spec_autograd import (
    speculative_verify,
    speculative_verify_pytorch_ref,
)

__all__ = [
    "speculative_verify",
    "speculative_verify_pytorch_ref",
]
