"""Attention mask utilities."""
from __future__ import annotations

import torch

__all__ = ["safe_mask_value", "_safe_mask_value"]


def safe_mask_value(dtype: torch.dtype) -> float:
    """Return safe additive mask value for the given dtype to avoid underflow/NaN."""
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


_safe_mask_value = safe_mask_value
