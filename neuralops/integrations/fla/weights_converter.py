#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FLA Checkpoint Weight Conversion Utility

Translates parameter keys and shapes between Flash Linear Attention (FLA)
naming conventions and NeuralOps canonical layer conventions.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping

import torch

logger = logging.getLogger(__name__)

__all__ = ["convert_fla_weights"]

# Forward mapping: FLA name -> NeuralOps name
FLA_TO_NEURALOPS_MAP: dict[str, str] = {
    "q_proj": "W_q",
    "k_proj": "W_k",
    "v_proj": "W_v",
    "g_proj": "W_g",
    "gate_proj": "W_g",
    "o_proj": "W_out",
    "out_proj": "W_out",
    "gk_proj": "W_gk",
    "gv_proj": "W_gv",
    "b_proj": "W_b",
    "beta_proj": "W_beta",
}

NEURALOPS_TO_FLA_MAP: dict[str, str] = {
    "W_q": "q_proj",
    "W_k": "k_proj",
    "W_v": "v_proj",
    "W_g": "g_proj",
    "W_out": "o_proj",
    "W_gk": "gk_proj",
    "W_gv": "gv_proj",
    "W_b": "b_proj",
    "W_beta": "beta_proj",
}


def convert_fla_weights(
    state_dict: Mapping[str, torch.Tensor],
    source_format: str = "fla",
    target_format: str = "neuralops",
) -> Dict[str, torch.Tensor]:
    """Convert weight tensor keys between FLA and NeuralOps conventions.

    Args:
        state_dict: Checkpoint state dictionary containing model weights.
        source_format: Source format convention ('fla' or 'neuralops').
        target_format: Target format convention ('neuralops' or 'fla').

    Returns:
        Converted state dictionary with updated parameter names.
    """
    source_norm = source_format.strip().lower()
    target_norm = target_format.strip().lower()

    if source_norm == target_norm:
        return dict(state_dict)

    converted: dict[str, torch.Tensor] = {}

    if source_norm == "fla" and target_norm == "neuralops":
        mapping = FLA_TO_NEURALOPS_MAP
    elif source_norm == "neuralops" and target_norm == "fla":
        mapping = NEURALOPS_TO_FLA_MAP
    else:
        raise ValueError(
            f"Unsupported conversion pair: source_format='{source_format}', target_format='{target_format}'."
        )

    for key, tensor in state_dict.items():
        new_key = key
        for src_name, dst_name in mapping.items():
            if f".{src_name}." in new_key or new_key.endswith(f".{src_name}") or new_key.startswith(f"{src_name}."):
                new_key = new_key.replace(f".{src_name}.", f".{dst_name}.")
                if new_key.endswith(f".{src_name}"):
                    new_key = new_key[: -len(src_name)] + dst_name
                if new_key.startswith(f"{src_name}."):
                    new_key = dst_name + new_key[len(src_name) :]
                break
        converted[new_key] = tensor

    return converted
