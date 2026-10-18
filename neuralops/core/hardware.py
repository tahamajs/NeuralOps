"""
Universal Hardware Probing Engine

Hardware topology detection and capability introspection across NVIDIA GPUs, AMD ROCm, Apple Silicon, and CPUs.

Author: Taha Majlesi
"""

from __future__ import annotations

import functools
import logging
import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger("neuralops.core.hardware")


class HardwareVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    ARM = "arm"
    GENERIC_CPU = "generic_cpu"
    UNKNOWN = "unknown"


class HardwareBackend(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    XPU = "xpu"
    CPU = "cpu"


@dataclass(frozen=True)
class HardwareTopology:
    """Comprehensive hardware topology and feature capability descriptor."""

    vendor: HardwareVendor
    backend: HardwareBackend
    device_name: str
    device_index: int = 0
    sm_major: int = 0
    sm_minor: int = 0
    compute_capability: str = "0.0"
    total_memory_bytes: int = 0
    max_shared_memory_bytes: int = 0
    warp_size: int = 32
    num_compute_units: int = 1

    # Feature flags
    supports_triton_jit: bool = False
    supports_cuda_cpp: bool = False
    supports_hip: bool = False
    supports_mps: bool = False
    supports_xpu: bool = False
    supports_tensor_cores: bool = False
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_fp8: bool = False

    # Supported dtypes
    supported_dtypes: tuple[torch.dtype, ...] = field(
        default_factory=lambda: (torch.float32, torch.float16)
    )

    @property
    def is_accelerated(self) -> bool:
        return self.backend != HardwareBackend.CPU

    @property
    def is_cuda_or_rocm(self) -> bool:
        return self.backend in (HardwareBackend.CUDA, HardwareBackend.ROCM)

    @property
    def is_apple_silicon(self) -> bool:
        return self.backend == HardwareBackend.MPS


# Architecture shared memory limits (bytes)
_ARCH_SHMEM_MAP = {
    (7, 0): 98304,
    (7, 5): 65536,
    (8, 0): 166912,
    (8, 6): 101376,
    (8, 9): 101376,
    (9, 0): 232448,
    (10, 0): 232448,
    (12, 0): 232448,
}


@functools.lru_cache(maxsize=32)
def probe_hardware(device: Optional[torch.device | str | int] = None) -> HardwareTopology:
    """
    Intelligently probe and return the hardware topology and capabilities for any device.
    Zero-crash guarantee across NVIDIA, AMD, Apple Silicon, Intel, and CPU.
    """
    # 1. Resolve device representation
    if device is None:
        device_str = os.environ.get("NEURALOPS_DEVICE", "auto").lower()
        if device_str != "auto":
            target_device = torch.device(device_str)
        elif torch.cuda.is_available():
            target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            target_device = torch.device("xpu")
        else:
            target_device = torch.device("cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    elif isinstance(device, int):
        target_device = torch.device(f"cuda:{device}")
    elif isinstance(device, torch.device):
        target_device = device
    else:
        target_device = torch.device("cpu")

    dev_type = target_device.type

    # 2. CPU Probe
    if dev_type == "cpu":
        cpu_name = platform.processor() or platform.machine() or "CPU"
        is_arm = "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower()
        vendor = HardwareVendor.ARM if is_arm else HardwareVendor.GENERIC_CPU
        try:
            num_cores = os.cpu_count() or 1
        except Exception:
            num_cores = 1

        supported_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]
        if hasattr(torch, "bfloat16"):
            supported_dtypes.append(torch.bfloat16)

        return HardwareTopology(
            vendor=vendor,
            backend=HardwareBackend.CPU,
            device_name=f"Host CPU ({cpu_name})",
            device_index=0,
            sm_major=0,
            sm_minor=0,
            compute_capability="0.0",
            total_memory_bytes=0,
            max_shared_memory_bytes=0,
            warp_size=1,
            num_compute_units=num_cores,
            supports_triton_jit=False,
            supports_cuda_cpp=False,
            supports_hip=False,
            supports_mps=False,
            supports_xpu=False,
            supports_tensor_cores=False,
            supports_fp16=True,
            supports_bf16=True,
            supports_tf32=False,
            supports_fp8=False,
            supported_dtypes=tuple(supported_dtypes),
        )

    # 3. Apple Silicon (MPS / Metal) Probe
    if dev_type == "mps":
        supports_bf16 = False
        try:
            # Test if bfloat16 works on this MPS runtime
            t = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            supports_bf16 = True
            del t
        except Exception:
            supports_bf16 = False

        supported_dtypes = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
        if supports_bf16:
            supported_dtypes.append(torch.bfloat16)

        return HardwareTopology(
            vendor=HardwareVendor.APPLE,
            backend=HardwareBackend.MPS,
            device_name=f"Apple Silicon Metal ({platform.processor() or 'M-Series'})",
            device_index=0,
            sm_major=0,
            sm_minor=0,
            compute_capability="mps",
            total_memory_bytes=0,
            max_shared_memory_bytes=32768,
            warp_size=32,
            num_compute_units=8,
            supports_triton_jit=False,
            supports_cuda_cpp=False,
            supports_hip=False,
            supports_mps=True,
            supports_xpu=False,
            supports_tensor_cores=False,
            supports_fp16=True,
            supports_bf16=supports_bf16,
            supports_tf32=False,
            supports_fp8=False,
            supported_dtypes=tuple(supported_dtypes),
        )

    # 4. Intel XPU Probe
    if dev_type == "xpu":
        dev_name = "Intel XPU / SYCL"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                dev_name = torch.xpu.get_device_name(target_device)
            except Exception:
                pass

        return HardwareTopology(
            vendor=HardwareVendor.INTEL,
            backend=HardwareBackend.XPU,
            device_name=dev_name,
            device_index=target_device.index or 0,
            sm_major=0,
            sm_minor=0,
            compute_capability="xpu",
            total_memory_bytes=0,
            max_shared_memory_bytes=65536,
            warp_size=32,
            num_compute_units=16,
            supports_triton_jit=False,
            supports_cuda_cpp=False,
            supports_hip=False,
            supports_mps=False,
            supports_xpu=True,
            supports_tensor_cores=False,
            supports_fp16=True,
            supports_bf16=True,
            supports_tf32=False,
            supports_fp8=False,
            supported_dtypes=(torch.float32, torch.float16, torch.bfloat16),
        )

    # 5. NVIDIA CUDA / AMD ROCm Probe
    if dev_type == "cuda":
        is_rocm = bool(hasattr(torch.version, "hip") and torch.version.hip is not None)
        backend = HardwareBackend.ROCM if is_rocm else HardwareBackend.CUDA
        vendor = HardwareVendor.AMD if is_rocm else HardwareVendor.NVIDIA

        idx = target_device.index if target_device.index is not None else torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        sm_major, sm_minor = int(cap[0]), int(cap[1])
        cc_str = f"{sm_major}.{sm_minor}"

        props = torch.cuda.get_device_properties(idx)
        total_mem = getattr(props, "total_memory", 0)
        num_sm = getattr(props, "multi_processor_count", 1)

        # Shared memory calculation
        shmem = getattr(props, "shared_memory_per_block_optin", None)
        if shmem is None or shmem <= 0:
            shmem = getattr(props, "shared_memory_per_block", None)
        if shmem is None or shmem <= 0:
            shmem = _ARCH_SHMEM_MAP.get((sm_major, sm_minor), 49152)

        # Acceleration flags
        supports_triton_jit = is_rocm or (sm_major >= 7)
        supports_cuda_cpp = not is_rocm
        supports_hip = is_rocm
        supports_tensor_cores = is_rocm or (sm_major >= 8) or (sm_major == 7 and sm_minor >= 5)

        if is_rocm:
            supports_bf16 = True
            supports_tf32 = False
        else:
            supports_bf16 = (sm_major >= 8) and bool(torch.cuda.is_bf16_supported())
            supports_tf32 = sm_major >= 8

        supports_fp8 = False
        if not is_rocm and sm_major >= 8 and (sm_minor >= 9 or sm_major >= 9):
            supports_fp8 = hasattr(torch, "float8_e4m3fn")
        elif is_rocm:
            supports_fp8 = hasattr(torch, "float8_e4m3fnuz")

        warp_size = int(getattr(props, "warp_size", 64 if is_rocm else 32))

        supported_dtypes_list = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
        if supports_bf16:
            supported_dtypes_list.append(torch.bfloat16)

        return HardwareTopology(
            vendor=vendor,
            backend=backend,
            device_name=dev_name,
            device_index=idx,
            sm_major=sm_major,
            sm_minor=sm_minor,
            compute_capability=cc_str,
            total_memory_bytes=total_mem,
            max_shared_memory_bytes=int(shmem),
            warp_size=warp_size,
            num_compute_units=num_sm,
            supports_triton_jit=supports_triton_jit,
            supports_cuda_cpp=supports_cuda_cpp,
            supports_hip=supports_hip,
            supports_mps=False,
            supports_xpu=False,
            supports_tensor_cores=supports_tensor_cores,
            supports_fp16=True,
            supports_bf16=supports_bf16,
            supports_tf32=supports_tf32,
            supports_fp8=supports_fp8,
            supported_dtypes=tuple(supported_dtypes_list),
        )

    # Fallback to CPU topology for unknown device
    return probe_hardware(torch.device("cpu"))
