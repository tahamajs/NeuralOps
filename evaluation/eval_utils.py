"""Evaluation utility helpers for batch filtering, logit/loss extraction, and cache eviction.

These functions provide low-level manipulation of PyTorch batches and model outputs
across standard Transformer model families.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default maximum entries per internal cache
DEFAULT_MAX_CACHE_SIZE: int = 16


def _filter_batch(
    batch: dict[str, torch.Tensor],
    model_family: str,
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Strip dictionary keys invalid for the target architecture while preserving labels.

    Args:
        batch: Dictionary of input tensors.
        model_family: Architecture family name (e.g. 'bert', 't5', 'gpt2').
        model: Instantiated PyTorch module whose forward signature is checked.

    Returns:
        Filtered copy of the batch containing only valid forward arguments.
    """
    family = model_family.lower()
    b = dict(batch)
    if "t5" in family:
        b.pop("token_type_ids", None)
    if "gpt2" in family:
        allowed = {"input_ids", "attention_mask", "labels"}
        b = {k: v for k, v in b.items() if k in allowed}

    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        valid = set(params.keys()) - {"self"}
        has_varkwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_varkwargs:
            b = {k: v for k, v in b.items() if k in valid}
    except (ValueError, TypeError):
        pass
    return b


def _extract_logits_and_loss(
    outputs: Any,
    num_labels: int = 2,
    labels: torch.Tensor | None = None,
    is_lm: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract logits and cross-entropy loss from arbitrary model output objects or tuples.

    Args:
        outputs: Output object returned by the model forward pass (e.g., HF ModelOutput or tuple).
        num_labels: Expected output class dimension for sequence classification.
        labels: Optional ground-truth target tensor.
        is_lm: True if language modeling task (preserves full vocab logits [B, T, V]).

    Returns:
        Tuple of (logits, loss). Either element may be None if unavailable.
    """
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss

    if hasattr(outputs, "logits"):
        logits = outputs.logits
        if not is_lm:
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            if logits.dim() == 2 and logits.shape[-1] != num_labels:
                if logits.shape[-1] > num_labels:
                    logits = logits[..., :num_labels]
    elif hasattr(outputs, "last_hidden_state"):
        hs = outputs.last_hidden_state
        logits = hs.mean(dim=1)[:, :num_labels]
        if labels is not None and logits.shape[-1] == num_labels and labels.dim() == 1:
            loss = F.cross_entropy(logits, labels)
    elif isinstance(outputs, (tuple, list)):
        first = outputs[0]
        if first.dim() == 3:
            logits = first if is_lm else first[:, -1, :]
            if not is_lm and logits.shape[-1] > num_labels:
                logits = logits[..., :num_labels]
        else:
            logits = first
            if logits.dim() == 0:
                loss = logits
                logits = None

    return logits, loss


def _evict_if_needed(cache: dict[Any, Any], max_size: int = DEFAULT_MAX_CACHE_SIZE) -> None:
    """Evict oldest entries in FIFO order from dictionary cache if size exceeds max_size.

    Args:
        cache: Dictionary acting as an LRU/FIFO cache.
        max_size: Maximum allowable number of entries.
    """
    if len(cache) <= max_size:
        return
    to_remove = list(cache.keys())[: len(cache) - max_size]
    for k in to_remove:
        del cache[k]


__all__ = [
    "DEFAULT_MAX_CACHE_SIZE",
    "_filter_batch",
    "_extract_logits_and_loss",
    "_evict_if_needed",
]
