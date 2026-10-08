"""Selective gradient checkpointing to balance memory and recomputation overhead."""
from __future__ import annotations
from typing import Callable, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = ["checkpoint_selective", "SelectiveCheckpointModule"]


def checkpoint_selective(
    function: Callable[..., Any],
    *args: Any,
    should_checkpoint: bool = True,
    use_reentrant: bool = False,
    **kwargs: Any,
) -> Any:
    """Selectively checkpoints execution based on runtime conditions (e.g. sequence length or memory threshold).

    Args:
        function: Forward callable.
        *args: Positional arguments.
        should_checkpoint: If True, uses torch.utils.checkpoint. If False, runs directly.
        use_reentrant: Whether to use reentrant autograd checkpointing.
        **kwargs: Keyword arguments passed to function.
    """
    if should_checkpoint:
        return checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
    return function(*args, **kwargs)


class SelectiveCheckpointModule(nn.Module):
    """Wraps an nn.Module and applies checkpointing selectively based on sequence length threshold."""

    def __init__(
        self,
        module: nn.Module,
        min_seqlen_threshold: int = 2048,
        use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.min_seqlen_threshold = min_seqlen_threshold
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        seqlen = x.shape[1] if x.dim() >= 2 else 0
        should_cp = self.training and (seqlen >= self.min_seqlen_threshold)
        if should_cp:
            def _custom_forward(*inputs):
                return self.module(*inputs, **kwargs)
            return checkpoint(_custom_forward, x, *args, use_reentrant=self.use_reentrant)
        return self.module(x, *args, **kwargs)
