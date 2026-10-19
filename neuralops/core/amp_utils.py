# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automatic Mixed Precision Utilities

Hardware-aware mixed precision helpers detecting GPU capability and managing optimal precision contexts.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import torch

logger = logging.getLogger(__name__)
_AMP_CONFIG = None


def detect_amp_config(force_dtype=None):
    """Returns dict: {dtype, use_scaler, use_tf32, reason}"""
    global _AMP_CONFIG
    if _AMP_CONFIG is not None and force_dtype is None:
        return _AMP_CONFIG

    if not torch.cuda.is_available():
        cfg = dict(dtype=torch.float32, use_scaler=False, use_tf32=False,
                   reason="CPU-only")
    else:
        cap = torch.cuda.get_device_capability(0)
        major = cap[0]
        name = torch.cuda.get_device_name(0)

        if major >= 8:
            cfg = dict(dtype=torch.bfloat16, use_scaler=False, use_tf32=True,
                       reason=f"Ada/Ampere {name} (cap {cap}) → bfloat16")
        elif major >= 7:
            cfg = dict(dtype=torch.float16, use_scaler=True, use_tf32=False,
                       reason=f"Turing/Volta {name} (cap {cap}) → float16 + scaler")
        else:
            cfg = dict(dtype=torch.float32, use_scaler=False, use_tf32=False,
                       reason=f"Pascal/older {name} (cap {cap}) → float32 (FP16 unstable)")

    if force_dtype is None:
        _AMP_CONFIG = cfg
    logger.info("[AMP] %s", cfg["reason"])
    return cfg


def maybe_half(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to FP16 ONLY on GPUs that benefit from it.
    On Pascal, keep FP32 (FP16 is slow and unstable here).
    On Turing, convert to FP16 (beneficial).
    On Ada/Ampere, keep as-is (autocast will use bf16 if needed).
    """
    if not x.is_cuda:
        return x
    cap = torch.cuda.get_device_capability(0)
    major = cap[0]
    if major >= 8:
        return x
    elif major >= 7:
        return x.half() if x.dtype == torch.float32 else x
    else:
        return x


def amp_context(device_type: str = "cuda"):
    """Context manager for autocast with auto-detected dtype."""
    cfg = detect_amp_config()
    if cfg["dtype"] == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type, dtype=cfg["dtype"])


def grad_scaler():
    """Returns GradScaler if needed, else None."""
    cfg = detect_amp_config()
    return torch.amp.GradScaler("cuda", enabled=True) if cfg["use_scaler"] else None


def setup_tf32():
    """Enable TF32 for FP32 ops on Ampere/Ada."""
    cfg = detect_amp_config()
    if cfg["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


__all__ = ["detect_amp_config", "maybe_half", "amp_context", "grad_scaler", "setup_tf32"]
