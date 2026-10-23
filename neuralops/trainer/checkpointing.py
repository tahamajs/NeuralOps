#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Selective activation checkpointing utilities for reducing training memory."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def checkpoint_selective(func, *args, should_checkpoint=True, **kwargs):
    if should_checkpoint:
        return checkpoint(func, *args, use_reentrant=False, **kwargs)
    return func(*args, **kwargs)

class SelectiveCheckpointModule(nn.Module):
    def __init__(self, module: nn.Module, min_seqlen_threshold: int = 0):
        super().__init__()
        self.module = module
        self.min_seqlen_threshold = min_seqlen_threshold

    def forward(self, *args, **kwargs):
        input_tensor = next(
            (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor) and value.dim() >= 2),
            None,
        )
        seqlen = input_tensor.shape[1] if input_tensor is not None else 0
        if seqlen >= self.min_seqlen_threshold:
            tensor_kwargs = {
                name: value for name, value in kwargs.items() if isinstance(value, torch.Tensor)
            }
            static_kwargs = {
                name: value for name, value in kwargs.items() if name not in tensor_kwargs
            }
            tensor_values = tuple(tensor_kwargs.values())

            def run_module(*checkpoint_args):
                positional_args = checkpoint_args[: len(args)]
                kwarg_values = checkpoint_args[len(args) :]
                recompute_kwargs = dict(static_kwargs)
                recompute_kwargs.update(zip(tensor_kwargs, kwarg_values, strict=True))
                return self.module(*positional_args, **recompute_kwargs)

            return checkpoint(run_module, *args, *tensor_values, use_reentrant=False)
        return self.module(*args, **kwargs)
