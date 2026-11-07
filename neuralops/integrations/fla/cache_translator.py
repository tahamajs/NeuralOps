#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cache Translation Layer for FLA (Fast Linear Attention) Integration.

Translates between Hugging Face Transformer KV cache representations and
FLA recurrent state representations in a strictly fail-closed manner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple
import torch

from neuralops.core.exceptions import IntegrationNotImplementedError

logger = logging.getLogger(__name__)


def _translate_hf_cache_to_fla_state(past_key_value: Any) -> Optional[torch.Tensor]:
    """
    Translate Hugging Face cache structure into an FLA recurrent associative state.

    Args:
        past_key_value: Incoming cache from Hugging Face model forward call.

    Returns:
        Recurrent state tensor for FLA, or None if no cache exists yet.

    Raises:
        IntegrationNotImplementedError: If the cache format is incompatible or cannot
            be translated without silent state loss.
    """
    if past_key_value is None:
        return None

    # If it is already an associative state tensor (B, H, Dk, Dv) or (B, H, D, D)
    if isinstance(past_key_value, torch.Tensor):
        if past_key_value.dim() in (3, 4):
            return past_key_value
        raise IntegrationNotImplementedError(
            f"Unsupported FLA recurrent state tensor dimensions: {past_key_value.shape}. "
            "Expected 3D or 4D state tensor."
        )

    # 2-tuple (K, V) of standard attention: shape (B, H, L, D)
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
        k, v = past_key_value
        if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
            # If sequence length is 0, start with empty state
            if k.size(-2) == 0:
                return None
            raise IntegrationNotImplementedError(
                "Converting historical 4D (K, V) softmax cache to FLA recurrent state matrix "
                "during active generation is not supported. Initialize model with clean state."
            )

    # DynamicCache or custom transformer cache objects
    type_name = type(past_key_value).__name__
    raise IntegrationNotImplementedError(
        f"Unsupported Hugging Face cache type '{type_name}' for FLA integration. "
        "Fail-closed per AGENTS.md §0.2 / §6.1."
    )


def _translate_fla_state_to_hf_cache(next_state: Any) -> Optional[Any]:
    """
    Translate FLA recurrent state to a format suitable for Hugging Face outputs.

    Args:
        next_state: Output state tensor from FLA inner layer.

    Returns:
        State tensor or tuple representation for Hugging Face downstream consumer.
    """
    if next_state is None:
        return None
    if isinstance(next_state, torch.Tensor):
        return next_state
    if isinstance(next_state, (tuple, list)):
        return next_state
    return next_state
