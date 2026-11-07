#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Serving Ecosystem Capability Matrix and Diagnostic Discovery.

Provides robust runtime capability detection across serving engines (vLLM, SGLang,
TGI, TRT-LLM) and JIT compiler targets (Triton, CUDA compute capabilities).
Fail-closed and safe across non-CUDA and non-Linux platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared capabilities and availability status of external serving frameworks.

    Attributes:
        name: Framework name ('vllm', 'sglang', 'tensorrt_llm', 'tgi').
        is_available: Whether the framework package is importable in current Python env.
        version: Package version string if available.
        supports_paged_attention: Whether paged KV-cache is supported.
        supports_speculative_decoding: Whether speculative drafting is supported.
        install_hint: Pip command to install or enable this backend.
    """

    name: str
    is_available: bool
    version: Optional[str]
    supports_paged_attention: bool
    supports_speculative_decoding: bool
    install_hint: str


def _safe_find_spec(module_name: str) -> bool:
    """Safely check if a module spec can be found without throwing unexpected errors."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError, TypeError):
        return False


def _safe_get_version(module_name: str) -> Optional[str]:
    """Safely retrieve module __version__ if importable."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", None)
    except (ImportError, AttributeError):
        return None


def probe_serving_ecosystem() -> Dict[str, Any]:
    """Detects serving engines, compilers, and hardware capabilities across the environment.

    Probes:
    - `vllm`: presence, version, whether CUDA ops are compiled.
    - `sglang`: presence, version, radix runtime support.
    - `tgi` (`text_generation_server`): presence, version, gRPC support.
    - `triton`: JIT capability, GPU target compute capability.

    Returns:
        Structured dictionary detailing availability and feature readiness.
    """
    results: Dict[str, Any] = {}

    # 1. vLLM detection
    vllm_avail = _safe_find_spec("vllm")
    vllm_ver = _safe_get_version("vllm") if vllm_avail else None
    vllm_cuda_ops = False
    if vllm_avail:
        # Check for compiled C/CUDA extensions
        # vLLM exposes compiled ops typically via vllm._C or vllm._custom_ops
        vllm_cuda_ops = (
            _safe_find_spec("vllm._C")
            or _safe_find_spec("vllm._custom_ops")
            or _safe_find_spec("vllm.vllm_ops")
        )

    results["vllm"] = {
        "is_available": vllm_avail,
        "version": vllm_ver,
        "has_cuda_ops": bool(vllm_cuda_ops),
    }

    # 2. SGLang detection
    sglang_avail = _safe_find_spec("sglang")
    sglang_ver = _safe_get_version("sglang") if sglang_avail else None
    sglang_radix = False
    if sglang_avail:
        # Check for radix runtime or sglang.srt
        sglang_radix = (
            _safe_find_spec("sglang.srt.mem_cache.radix_cache")
            or _safe_find_spec("sglang.srt.radix_cache")
            or _safe_find_spec("sglang.srt")
        )

    results["sglang"] = {
        "is_available": sglang_avail,
        "version": sglang_ver,
        "supports_radix": bool(sglang_radix),
    }

    # 3. TGI (text_generation_server) detection
    tgi_avail = _safe_find_spec("text_generation_server")
    tgi_ver = _safe_get_version("text_generation_server") if tgi_avail else None
    tgi_grpc = False
    if tgi_avail:
        tgi_grpc = (
            _safe_find_spec("text_generation_server.pb")
            or _safe_find_spec("text_generation_server.server")
        )

    results["tgi"] = {
        "is_available": tgi_avail,
        "version": tgi_ver,
        "has_grpc": bool(tgi_grpc),
    }

    # 4. Triton detection & GPU target compute capability
    triton_avail = _safe_find_spec("triton")
    triton_ver = _safe_get_version("triton") if triton_avail else None
    triton_jit_capable = False
    if triton_avail:
        try:
            import triton

            triton_jit_capable = hasattr(triton, "jit")
        except (ImportError, AttributeError):
            triton_jit_capable = False

    target_compute_capability: Optional[Tuple[int, int]] = None
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            target_compute_capability = torch.cuda.get_device_capability()
        except (RuntimeError, AssertionError):
            target_compute_capability = None

    results["triton"] = {
        "is_available": triton_avail,
        "version": triton_ver,
        "jit_capable": bool(triton_jit_capable),
        "target_compute_capability": target_compute_capability,
    }

    return results


def query_serving_capabilities() -> Dict[str, BackendCapabilities]:
    """Inspects installed libraries and returns capability descriptor for each serving backend."""
    capabilities: Dict[str, BackendCapabilities] = {}

    ecosystem = probe_serving_ecosystem()

    # 1. vLLM
    vllm_info = ecosystem["vllm"]
    capabilities["vllm"] = BackendCapabilities(
        name="vllm",
        is_available=vllm_info["is_available"],
        version=vllm_info["version"],
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install vllm",
    )

    # 2. SGLang
    sglang_info = ecosystem["sglang"]
    capabilities["sglang"] = BackendCapabilities(
        name="sglang",
        is_available=sglang_info["is_available"],
        version=sglang_info["version"],
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install sglang",
    )

    # 3. TensorRT-LLM
    trt_avail = _safe_find_spec("tensorrt_llm")
    trt_ver = _safe_get_version("tensorrt_llm") if trt_avail else None
    capabilities["tensorrt_llm"] = BackendCapabilities(
        name="tensorrt_llm",
        is_available=trt_avail,
        version=trt_ver,
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install tensorrt_llm",
    )

    # 4. TGI
    tgi_info = ecosystem["tgi"]
    capabilities["tgi"] = BackendCapabilities(
        name="tgi",
        is_available=tgi_info["is_available"],
        version=tgi_info["version"],
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install text_generation_server",
    )

    # 5. Native NeuralOps Serving
    capabilities["neuralops_native"] = BackendCapabilities(
        name="neuralops_native",
        is_available=True,
        version="2.0.3",
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="Already included in NeuralOps",
    )

    return capabilities


__all__ = [
    "BackendCapabilities",
    "probe_serving_ecosystem",
    "query_serving_capabilities",
]
