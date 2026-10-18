"""
State Space Model Update Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.state_update.state_autograd import (
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
