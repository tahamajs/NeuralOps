#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Preflight and Device Normalization for NeuralOps CLI.

Enforces strict fail-closed acceleration requirements without silent CPU fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from .errors import BackendUnavailableError


@dataclass(frozen=True)
class DeviceInfo:
    """Canonical hardware and device execution context."""

    requested: str
    resolved: str
    kind: str
    index: int | None
    name: str | None
    driver_version: str | None
    cuda_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "kind": self.kind,
            "index": self.index,
            "name": self.name,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
        }


def resolve_device(requested: str | None, *, allow_auto: bool = True) -> DeviceInfo:
    """Resolves and validates execution hardware with fail-closed semantics.

    Args:
        requested: Requested device string (e.g. 'cuda', 'cuda:0', 'mps', 'cpu', 'auto').
        allow_auto: Whether 'auto' resolution is permitted.

    Returns:
        DeviceInfo dataclass representing verified execution hardware.

    Raises:
        BackendUnavailableError: If an explicitly requested accelerator is unavailable.
        ValueError: If the device specification string is invalid.
    """
    req = (requested or "auto").lower().strip()

    if req == "auto":
        if not allow_auto:
            raise BackendUnavailableError("Automatic device resolution is prohibited for this command.")
        if torch.cuda.is_available():
            req = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"

    if req == "cuda":
        req = "cuda:0"

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                f"CUDA device '{req}' was requested, but CUDA is unavailable on this host. "
                "Failing closed to prevent invalid latency or throughput measurement.",
                hint="Verify NVIDIA driver installation or run 'neuralops system verify --device cuda'.",
            )
        try:
            index = int(req.split(":", 1)[1]) if ":" in req else 0
        except ValueError as e:
            raise ValueError(f"Invalid CUDA device index in '{req}'") from e

        num_devices = torch.cuda.device_count()
        if index >= num_devices or index < 0:
            raise BackendUnavailableError(
                f"Requested CUDA index {index} exceeds available devices (total: {num_devices}).",
                hint=f"Use a valid device index between 0 and {max(0, num_devices - 1)}.",
            )

        dev_name = torch.cuda.get_device_name(index)
        cuda_ver = torch.version.cuda
        return DeviceInfo(
            requested=requested or "auto",
            resolved=f"cuda:{index}",
            kind="cuda",
            index=index,
            name=dev_name,
            driver_version=None,
            cuda_version=cuda_ver,
        )

    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise BackendUnavailableError(
                "Apple Silicon MPS device was requested, but MPS acceleration is unavailable.",
                hint="Verify macOS version and PyTorch arm64 installation.",
            )
        return DeviceInfo(
            requested=requested or "auto",
            resolved="mps",
            kind="mps",
            index=None,
            name="Apple Silicon MPS",
            driver_version=None,
            cuda_version=None,
        )

    if req == "cpu":
        return DeviceInfo(
            requested=requested or "auto",
            resolved="cpu",
            kind="cpu",
            index=None,
            name="CPU Host",
            driver_version=None,
            cuda_version=None,
        )

    raise ValueError(f"Unsupported device string '{requested}'. Allowed: 'cpu', 'cuda', 'cuda:N', 'mps', 'auto'.")
