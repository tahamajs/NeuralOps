#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Self-Diagnosing Error Advisor

Intercepts common runtime and configuration exceptions, performs fuzzy matching
against registered attention mechanisms and compressors, and produces actionable
recommendations for developers.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "diagnose_error",
    "find_closest_match",
    "levenshtein_distance",
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_closest_match(target: str, candidates: Iterable[str]) -> tuple[str, int]:
    """Find the candidate string with the smallest Levenshtein distance."""
    target_clean = str(target).strip("'\"").lower()
    best_candidate = ""
    min_dist = float("inf")
    for cand in candidates:
        cand_str = str(cand).lower()
        dist = levenshtein_distance(target_clean, cand_str)
        if dist < min_dist:
            min_dist = dist
            best_candidate = cand
    return best_candidate, int(min_dist)


COMMON_ALIASES: tuple[str, ...] = (
    "linear_attention",
    "gated_linear_attention",
    "flash_attention",
    "flash_attention_2",
    "flash_attention_3",
    "mamba",
    "mamba2",
    "rwkv6",
    "retnet",
)


def _get_registry_keys() -> list[str]:
    """Lazily load keys from ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, and common aliases."""
    keys: list[str] = list(COMMON_ALIASES)
    try:
        from neuralops import ATTENTION_REGISTRY
        keys.extend(list(ATTENTION_REGISTRY.keys()))
    except Exception:
        # allowed-swallow: fallback registry keys
        pass
    try:
        from neuralops import COMPRESSOR_REGISTRY
        keys.extend(list(COMPRESSOR_REGISTRY.keys()))
    except Exception:
        # allowed-swallow: fallback registry keys
        pass
    return list(dict.fromkeys(keys))


def diagnose_error(
    exception: Exception,
    *,
    context: Optional[dict[str, Any]] = None,
    auto_format: bool = True,
) -> str:
    """Analyze an exception and generate an actionable diagnostic message.

    Args:
        exception: The raised exception.
        context: Optional diagnostic context (e.g. subcommand, arguments, tensors).
        auto_format: Whether to format output with standard DIAGNOSIS and RECOMMENDATION blocks.

    Returns:
        Human-readable diagnostic report with actionable recommendations.
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception)
    full_msg = f"{exc_type}: {exc_msg}"

    diagnosis = ""
    recommendation = ""

    # 1. IntegrationNotImplementedError
    if "IntegrationNotImplementedError" in exc_type:
        diagnosis = f"Integration feature unavailable: {exc_msg}"
        recommendation = (
            "This integration is scheduled for Roadmap v2.0.0 (see ROADMAP.md). "
            "For active inference serving, use PyTorch native execution or export to safetensors."
        )

    # 2. KeyError (Registry lookups)
    elif isinstance(exception, KeyError):
        raw_key = str(exception).strip("'\"")
        candidates = _get_registry_keys()
        if candidates:
            closest, dist = find_closest_match(raw_key, candidates)
            diagnosis = f"Registry key '{raw_key}' was not found."
            if closest and dist <= 5:
                recommendation = (
                    f"Did you mean '{closest}'? "
                    "Run 'neuralops list' to see all 131 attention mechanisms and 72 compressors."
                )
            else:
                recommendation = "Run 'neuralops list' to inspect available registered architectures and compressors."
        else:
            diagnosis = f"Key '{raw_key}' not found."
            recommendation = "Verify the specified name against registry definitions."

    # 3. OutOfMemoryError / CUDA OOM
    elif "OutOfMemoryError" in exc_type or "out of memory" in exc_msg.lower():
        diagnosis = "CUDA Out Of Memory (OOM) encountered during execution."
        recommendation = (
            "Set NEURALOPS_AUTOGRAD_CHECKPOINT=1 to enable activation checkpointing, "
            "reduce sequence length or batch size, or switch to FP16/BF16 mixed precision."
        )

    # 4. Shape or Dimension Mismatch
    elif "mismatch" in exc_msg.lower() and ("shape" in exc_msg.lower() or "size" in exc_msg.lower() or "dimension" in exc_msg.lower()):
        diagnosis = f"Tensor shape or dimension mismatch: {exc_msg}"
        recommendation = (
            "Verify input tensor shapes against the operator contract. "
            "Ensure head dimension D and sequence length L match across Q, K, and V."
        )

    # 5. Default Fallback
    else:
        diagnosis = f"Execution failed with {exc_type}: {exc_msg}"
        recommendation = "Review the execution parameters or run with --verbose to inspect detailed telemetry."

    if not auto_format:
        return f"{diagnosis}\n{recommendation}"

    return (
        f"[bold red]DIAGNOSIS:[/] {diagnosis}\n"
        f"[bold cyan]RECOMMENDATION:[/] {recommendation}"
    )
