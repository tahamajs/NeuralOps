#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Ecosystem Capability Matrix and Diagnostic Discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Dict, Optional


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


def query_serving_capabilities() -> Dict[str, BackendCapabilities]:
    """Inspects installed libraries and returns capability descriptor for each serving backend."""
    capabilities: Dict[str, BackendCapabilities] = {}

    # 1. vLLM
    vllm_spec = importlib.util.find_spec("vllm")
    vllm_avail = vllm_spec is not None
    vllm_ver = None
    if vllm_avail:
        try:
            import vllm

            vllm_ver = getattr(vllm, "__version__", None)
        except Exception:
            pass

    capabilities["vllm"] = BackendCapabilities(
        name="vllm",
        is_available=vllm_avail,
        version=vllm_ver,
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install vllm",
    )

    # 2. SGLang
    sglang_spec = importlib.util.find_spec("sglang")
    sglang_avail = sglang_spec is not None
    sglang_ver = None
    if sglang_avail:
        try:
            import sglang

            sglang_ver = getattr(sglang, "__version__", None)
        except Exception:
            pass

    capabilities["sglang"] = BackendCapabilities(
        name="sglang",
        is_available=sglang_avail,
        version=sglang_ver,
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install sglang",
    )

    # 3. TensorRT-LLM
    trt_spec = importlib.util.find_spec("tensorrt_llm")
    trt_avail = trt_spec is not None
    trt_ver = None
    if trt_avail:
        try:
            import tensorrt_llm

            trt_ver = getattr(tensorrt_llm, "__version__", None)
        except Exception:
            pass

    capabilities["tensorrt_llm"] = BackendCapabilities(
        name="tensorrt_llm",
        is_available=trt_avail,
        version=trt_ver,
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="pip install tensorrt_llm",
    )

    # 4. Native NeuralOps Serving
    capabilities["neuralops_native"] = BackendCapabilities(
        name="neuralops_native",
        is_available=True,
        version="2.0.2",
        supports_paged_attention=True,
        supports_speculative_decoding=True,
        install_hint="Already included in NeuralOps",
    )

    return capabilities
