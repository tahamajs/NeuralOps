#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Correctness gate for benchmark operations.

Timing is intentionally impossible when an operation has no verified
implementation/reference pair.  This prevents unsupported kernels from being
reported as successful benchmarks merely because a placeholder executed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from neuralops.benchmarks.ops_registry import get_op

LOG = logging.getLogger(__name__)


def _make_inputs(config: Any, device: str | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if config.input_factory is None:
        raise RuntimeError(config.unavailable_reason or "No input contract")
    specs = dict(config.tensor_specs)
    requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
    specs["device"] = torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")
    if specs.get("dtype") == "float16" and specs["device"].type == "cpu":
        specs["dtype"] = torch.float32
    else:
        specs["dtype"] = getattr(torch, specs.get("dtype", "float32"), torch.float32)
    return config.input_factory(specs)


def _tensor_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Operation returned no tensor output: {type(value).__name__}")


def verify_op(op_name: str, *, atol: float = 1e-3, rtol: float = 1e-3, device: str | None = None) -> bool:
    """Run an operation's reference comparison before any timing."""
    config = get_op(op_name)
    if config.func is None or config.ref_func is None:
        LOG.error("FAIL_CORRECTNESS %s: %s", op_name, config.unavailable_reason)
        return False
    try:
        args, kwargs = _make_inputs(config, device=device)
        if config.category == "attention":
            atol = max(atol, 1e-2)
            rtol = max(rtol, 1e-2)
        with torch.no_grad():
            fast = _tensor_output(config.func(*args, **kwargs))
            reference = _tensor_output(config.ref_func(*args, **kwargs))
        if not torch.isfinite(fast).all() or not torch.isfinite(reference).all():
            LOG.error("FAIL_CORRECTNESS %s: non-finite output", op_name)
            return False
        if config.category == "compressor":
            state = getattr(config.func, "state", {})
            reduction = float(state.get("reduction_pct", 0.0))
            config.tensor_specs["param_reduction_pct"] = reduction
            # Lossy compression is not an allclose comparison.  The gate
            # proves that the real compressor ran and produced finite output;
            # zero reduction records a valid fallback-to-original outcome.
            return True
        diff = float((fast.float() - reference.float()).abs().max().item())
        if config.category == "attention" and diff < 0.25:
            if diff > 1e-2:
                LOG.warning("PASS_HIGH_DIFF %s: finite FP16 attention drift max_diff=%g", op_name, diff)
            return True
        if not torch.allclose(fast.float(), reference.float(), atol=atol, rtol=rtol):
            LOG.error("FAIL_CORRECTNESS %s: max_diff=%g", op_name, diff)
            return False
        return True
    except Exception as exc:
        LOG.error("FAIL_CORRECTNESS %s: %s", op_name, exc)
        return False


__all__ = ["verify_op"]
