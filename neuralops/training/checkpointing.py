# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Training Checkpointing Module

Provides crash-safe model checkpoint saving, state resumption, and artifact management.
"""

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
        seqlen = 0
        if len(args) > 0 and hasattr(args[0], "shape") and args[0].dim() >= 2:
            seqlen = args[0].shape[1]
        if seqlen >= self.min_seqlen_threshold:
            return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)
