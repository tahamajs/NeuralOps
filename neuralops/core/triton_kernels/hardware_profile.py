"""
Hardware Capability Profiler for OpenAI Triton & PyTorch Acceleration Backends.

Provides cached, cross-vendor introspection for:
  - NVIDIA GPUs (Pascal SM 6.1 through Blackwell SM 10.0 / 12.0)
  - AMD ROCm (MI200/MI300 CDNA architectures)
  - Apple Silicon (mps)
  - Intel GPUs (xpu)
  - CPU
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger("HardwareProfile")

BackendType = Literal["cuda", "rocm", "xpu", "mps", "cpu"]


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable hardware capability profile for Triton JIT execution and memory safety."""

    backend: BackendType
    device_name: str
    sm_major: int
    sm_minor: int
    max_shared_memory_bytes: int
    supports_triton_jit: bool
    supports_tensor_core_dot: bool
    supports_bf16: bool
    supports_tf32: bool
    warp_size: int


# Default architecture-based shared memory limits (in bytes)
# Used when PyTorch device properties do not expose opt-in shared memory directly
_ARCH_DEFAULT_SHARED_MEMORY = {
    # Maxwell / Pascal (SM 5.x / 6.x): 48 KB per block
    (5, 0): 49152,
    (5, 2): 49152,
    (6, 0): 65536,
    (6, 1): 49152,
    # Volta (SM 7.0): up to 96 KB opt-in (default 48 KB)
    (7, 0): 98304,
    # Turing (SM 7.5: T4, RTX 2080): 64 KB per block
    (7, 5): 65536,
    # Ampere datacenter (SM 8.0: A100): 163 KB (166,912 bytes opt-in)
    (8, 0): 166912,
    # Ampere client (SM 8.6: RTX 3080/3090): 99 KB (101,376 bytes opt-in)
    (8, 6): 101376,
    # Ada Lovelace (SM 8.9: RTX 4070/4080/4090, L40): 99 KB (101,376 bytes opt-in)
    (8, 9): 101376,
    # Hopper (SM 9.0: H100, H200): 227 KB (232,448 bytes opt-in)
    (9, 0): 232448,
    # Blackwell (SM 10.0 / 12.0: B200): 227 KB+
    (10, 0): 232448,
    (12, 0): 232448,
}


@functools.lru_cache(maxsize=32)
def get_device_hardware_profile(device: torch.device | str | int | None = None) -> HardwareProfile:
    """Inspect and cache the HardwareProfile for the specified device."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")

    device_type = device.type

    # 1. Non-CUDA / CPU / MPS / XPU Branches
    if device_type == "cpu":
        return HardwareProfile(
            backend="cpu",
            device_name="CPU",
            sm_major=0,
            sm_minor=0,
            max_shared_memory_bytes=0,
            supports_triton_jit=False,
            supports_tensor_core_dot=False,
            supports_bf16=True,
            supports_tf32=False,
            warp_size=1,
        )

    if device_type == "mps":
        return HardwareProfile(
            backend="mps",
            device_name="Apple Silicon (MPS)",
            sm_major=0,
            sm_minor=0,
            max_shared_memory_bytes=32768,
            supports_triton_jit=False,
            supports_tensor_core_dot=False,
            supports_bf16=True,
            supports_tf32=False,
            warp_size=32,
        )

    if device_type == "xpu":
        dev_name = "Intel XPU"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                dev_name = torch.xpu.get_device_name(device)
            except Exception:
                pass
        return HardwareProfile(
            backend="xpu",
            device_name=dev_name,
            sm_major=0,
            sm_minor=0,
            max_shared_memory_bytes=65536,
            supports_triton_jit=False,
            supports_tensor_core_dot=False,
            supports_bf16=True,
            supports_tf32=False,
            warp_size=32,
        )

    # 2. CUDA / ROCm Branches
    if device_type == "cuda":
        # Check for AMD ROCm
        is_rocm = bool(hasattr(torch.version, "hip") and torch.version.hip is not None)
        backend: BackendType = "rocm" if is_rocm else "cuda"

        device_idx = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        cap = torch.cuda.get_device_capability(device_idx)
        sm_major, sm_minor = int(cap[0]), int(cap[1])

        props = torch.cuda.get_device_properties(device_idx)

        # Shared Memory Calculation
        # Prefer opt-in shared memory per block if available in PyTorch, else fallback to standard arch mapping
        shmem = getattr(props, "shared_memory_per_block_optin", None)
        if shmem is None or shmem <= 0:
            shmem = getattr(props, "shared_memory_per_block", None)

        if shmem is None or shmem <= 0:
            shmem = _ARCH_DEFAULT_SHARED_MEMORY.get((sm_major, sm_minor))
            if shmem is None:
                if sm_major >= 9:
                    shmem = 232448
                elif sm_major >= 8:
                    shmem = 101376
                elif sm_major == 7:
                    shmem = 65536 if sm_minor >= 5 else 98304
                else:
                    shmem = 49152

        # Capability Flags
        # Basic Triton JIT requires SM >= 7.0 on CUDA (or any ROCm device)
        supports_triton_jit = is_rocm or (sm_major >= 7)

        # Tensor Core Dot support:
        # Volta (SM 7.0) had HMMA instructions, Turing (SM 7.5) and Ampere+ (SM 8.0+) have full mma.sync
        # Triton tl.dot requires SM >= 8.0 for modern mma, or SM >= 7.5 for Turing
        supports_tensor_core_dot = is_rocm or (sm_major >= 8) or (sm_major == 7 and sm_minor >= 5)

        # BF16 hardware support: Ampere (SM 8.0) and above on NVIDIA
        # Note: torch.cuda.is_bf16_supported() can report True on older drivers if emulation exists;
        # strictly enforce SM >= 8.0 for native hardware BF16!
        if is_rocm:
            supports_bf16 = True
        else:
            supports_bf16 = (sm_major >= 8) and bool(torch.cuda.is_bf16_supported())

        # TF32 hardware support: Ampere (SM 8.0) and above
        supports_tf32 = (not is_rocm) and (sm_major >= 8)

        # Warp size: 64 on AMD ROCm, 32 on NVIDIA CUDA
        warp_size = int(getattr(props, "warp_size", 64 if is_rocm else 32))

        return HardwareProfile(
            backend=backend,
            device_name=device_name,
            sm_major=sm_major,
            sm_minor=sm_minor,
            max_shared_memory_bytes=int(shmem),
            supports_triton_jit=bool(supports_triton_jit),
            supports_tensor_core_dot=bool(supports_tensor_core_dot),
            supports_bf16=bool(supports_bf16),
            supports_tf32=bool(supports_tf32),
            warp_size=warp_size,
        )

    # Fallback default
    return HardwareProfile(
        backend="cpu",
        device_name="Unknown Device",
        sm_major=0,
        sm_minor=0,
        max_shared_memory_bytes=0,
        supports_triton_jit=False,
        supports_tensor_core_dot=False,
        supports_bf16=False,
        supports_tf32=False,
        warp_size=32,
    )
