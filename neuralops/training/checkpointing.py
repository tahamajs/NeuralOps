"""Checkpointing compatibility module."""
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
