"""Cache manipulation utilities for autoregressive generation and streaming states."""

from __future__ import annotations

from typing import Any
import torch


def read_cache(
    past_key_value: Any, layer_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, Any | None]:
    """Extract key, value, sequence length, and optional SSM/recurrent state from past_key_value cache."""
    if past_key_value is None:
        return None, None, 0, None

    if isinstance(past_key_value, (tuple, list)):
        if len(past_key_value) == 0:
            return None, None, 0, None
        first = past_key_value[0]
        if isinstance(first, (tuple, list)):
            if layer_idx < len(past_key_value):
                entry = past_key_value[layer_idx]
                k, v = entry[0], entry[1]
                m = entry[2] if len(entry) > 2 else None
                return k, v, k.shape[-2], m
            return None, None, 0, None
        else:
            k, v = past_key_value[0], past_key_value[1]
            m = past_key_value[2] if len(past_key_value) > 2 else None
            return k, v, k.shape[-2], m

    if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
        if layer_idx < len(past_key_value.key_cache):
            k = past_key_value.key_cache[layer_idx]
            v = (
                past_key_value.value_cache[layer_idx]
                if hasattr(past_key_value, "value_cache") and layer_idx < len(past_key_value.value_cache)
                else None
            )
            m = None
            if hasattr(past_key_value, "mamba_state") and past_key_value.mamba_state:
                if layer_idx < len(past_key_value.mamba_state):
                    m = past_key_value.mamba_state[layer_idx]
            if k is not None:
                return k, v, k.shape[-2], m

    return None, None, 0, None


def write_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    mamba_state: Any | None,
    past_key_value: Any,
    layer_idx: int,
) -> Any:
    """Update or construct the KV and recurrent state cache for autoregressive decoding."""
    if past_key_value is not None and hasattr(past_key_value, "key_cache"):
        if layer_idx < len(past_key_value.key_cache):
            past_key_value.key_cache[layer_idx] = k
            past_key_value.value_cache[layer_idx] = v
        else:
            past_key_value.key_cache.append(k)
            past_key_value.value_cache.append(v)

        if not hasattr(past_key_value, "mamba_state"):
            past_key_value.mamba_state = []
        if layer_idx < len(past_key_value.mamba_state):
            past_key_value.mamba_state[layer_idx] = mamba_state
        else:
            past_key_value.mamba_state.append(mamba_state)
        return past_key_value

    if mamba_state is not None:
        return (k, v, mamba_state)
    return (k, v)


# Backward compatibility aliases
_read_cache = read_cache
_write_cache = write_cache
