"""Activation & Gradient Checkpointing module with automated injection for ultra-long context."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def checkpoint_selective(func: Callable, *args: Any, should_checkpoint: bool = True, **kwargs: Any) -> Any:
    """Selectively applies non-reentrant activation checkpointing."""
    if should_checkpoint:
        return checkpoint(func, *args, use_reentrant=False, **kwargs)
    return func(*args, **kwargs)


class SelectiveCheckpointModule(nn.Module):
    """Wraps an nn.Module with dynamic seqlen-thresholded activation checkpointing."""

    def __init__(self, module: nn.Module, min_seqlen_threshold: int = 0) -> None:
        super().__init__()
        self.module = module
        self.min_seqlen_threshold = min_seqlen_threshold

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        seqlen = 0
        if len(args) > 0 and hasattr(args[0], "shape") and args[0].dim() >= 2:
            seqlen = args[0].shape[1]
        elif "hidden_states" in kwargs and hasattr(kwargs["hidden_states"], "shape"):
            seqlen = kwargs["hidden_states"].shape[1]

        if seqlen >= self.min_seqlen_threshold and self.training:
            return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)


def inject_activation_checkpointing(
    model: nn.Module,
    layer_types: tuple[type, ...] | None = None,
    min_seqlen_threshold: int = 0,
    check_name_substrings: tuple[str, ...] = ("block", "layer", "decoder_layer"),
) -> int:
    """Automatically replaces target transformer blocks with SelectiveCheckpointModule.

    Args:
        model: Target PyTorch model.
        layer_types: Tuple of Module classes to wrap.
        min_seqlen_threshold: Minimum sequence length to trigger recomputation during forward.
        check_name_substrings: Name substrings to target if layer_types is None.

    Returns:
        Number of wrapped layers.
    """
    wrapped_count = 0

    for name, child in list(model.named_children()):
        should_wrap = False
        if layer_types is not None and isinstance(child, layer_types):
            should_wrap = True
        elif layer_types is None and any(sub in name.lower() for sub in check_name_substrings):
            # Check if child is a container or composite block
            if len(list(child.children())) > 0 and not isinstance(child, SelectiveCheckpointModule):
                should_wrap = True

        if should_wrap and not isinstance(child, SelectiveCheckpointModule):
            setattr(model, name, SelectiveCheckpointModule(child, min_seqlen_threshold=min_seqlen_threshold))
            wrapped_count += 1
        else:
            wrapped_count += inject_activation_checkpointing(
                child,
                layer_types=layer_types,
                min_seqlen_threshold=min_seqlen_threshold,
                check_name_substrings=check_name_substrings,
            )

    return wrapped_count


__all__ = [
    "checkpoint_selective",
    "SelectiveCheckpointModule",
    "inject_activation_checkpointing",
]
