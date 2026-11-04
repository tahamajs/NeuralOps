#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Host Platform & Execution Environment Profiler

Freezes and inspects the hardware, OS, CUDA runtime, and software dependencies
for reproducible benchmark provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _get_cpu_info() -> dict[str, Any]:
    """Inspect host CPU architecture, core counts, and frequency."""
    cpu_data: dict[str, Any] = {
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "physical_cores": None,
        "logical_threads": os.cpu_count() or 1,
        "base_frequency_mhz": None,
    }
    try:
        import psutil

        cpu_data["physical_cores"] = psutil.cpu_count(logical=False)
        cpu_data["logical_threads"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq and freq.current:
            cpu_data["base_frequency_mhz"] = round(float(freq.current), 2)
    except Exception:
        pass

    return cpu_data


def _get_memory_info() -> dict[str, Any]:
    """Inspect total host system RAM and swap."""
    mem_data: dict[str, Any] = {
        "total_ram_mb": None,
        "available_ram_mb": None,
        "total_swap_mb": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        mem_data["total_ram_mb"] = round(float(vm.total) / (1024.0 * 1024.0), 2)
        mem_data["available_ram_mb"] = round(float(vm.available) / (1024.0 * 1024.0), 2)
        sm = psutil.swap_memory()
        mem_data["total_swap_mb"] = round(float(sm.total) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    return mem_data


def _get_gpu_devices() -> list[dict[str, Any]]:
    """Enumerate attached GPU accelerator devices."""
    devices: list[dict[str, Any]] = []

    # 1. Try PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                dev_entry: dict[str, Any] = {
                    "index": idx,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_vram_mb": round(float(props.total_memory) / (1024.0 * 1024.0), 2),
                    "pci_bus_id": None,
                }
                devices.append(dev_entry)
            if devices:
                return devices
    except Exception:
        pass

    # 2. Try nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, shell=False, text=True, timeout=2).strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "compute_capability": None,
                        "total_vram_mb": float(parts[2]),
                        "pci_bus_id": parts[3],
                    }
                )
        if devices:
            return devices
    except Exception:
        pass

    return devices


def _get_package_inventory_hash() -> str:
    """Generate deterministic SHA-256 hash of installed Python packages."""
    try:
        import importlib.metadata

        dists = sorted([f"{d.name}=={d.version}" for d in importlib.metadata.distributions()])
        return hashlib.sha256("\n".join(dists).encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


def get_host_environment() -> dict[str, Any]:
    """
    Construct the complete canonical host environment dictionary.
    Guaranteed to be 100% JSON-serializable.
    """
    # PyTorch and CUDA versions
    pytorch_version = None
    cuda_runtime_version = None
    cudnn_version = None

    try:
        import torch

        pytorch_version = str(torch.__version__)
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            cuda_runtime_version = str(torch.version.cuda)
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())
    except Exception:
        pass

    # NVIDIA Driver version
    nvidia_driver_version = None
    try:
        cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        out = subprocess.check_output(cmd, shell=False, text=True, timeout=2).strip()
        if out:
            nvidia_driver_version = out.splitlines()[0].strip()
    except Exception:
        pass

    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_interpreter": sys.executable,
        "pytorch_version": pytorch_version,
        "cuda_runtime_version": cuda_runtime_version,
        "cudnn_version": cudnn_version,
        "nvidia_driver_version": nvidia_driver_version,
        "cpu_info": _get_cpu_info(),
        "memory_info": _get_memory_info(),
        "gpu_devices": _get_gpu_devices(),
        "package_inventory_hash": _get_package_inventory_hash(),
    }

    # Ensure JSON serializability
    json.dumps(env)
    return env
