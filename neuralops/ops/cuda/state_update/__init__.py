#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
State Space Model Update Hardware Operators.
"""

from neuralops.ops.cuda.state_update.state_autograd import (
    recurrent_linear_step,
    recurrent_linear_step_pytorch_ref,
)

state_update = recurrent_linear_step
state_update_pytorch_ref = recurrent_linear_step_pytorch_ref

__all__ = [
    "recurrent_linear_step",
    "recurrent_linear_step_pytorch_ref",
    "state_update",
    "state_update_pytorch_ref",
]
