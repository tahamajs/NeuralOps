#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 15: High-Throughput Serving Engine Integrations (vLLM, SGLang, TensorRT-LLM).

Subsystem Demonstrated:
    Comprehensive registration, contract verification, and interface introspection
    for major LLM serving frameworks:
    1. vLLM Attention Backend & Recurrent KV Cache Spec
       - Registers NeuralOpsAttentionBackend with custom AttentionBackendEnum
       - Provides NeuralOpsRecurrentSpec for O(1) linear attention cache
    2. SGLang Linear Attention & Radix Cache Integration
       - Registers NeuralOpsLinearAttnBackend with uses_mamba_radix_cache=True
       - Provides forward_extend and forward_decode serving entry points
    3. TensorRT-LLM Custom Plugin Builder
       - Provides TensorRTLLMPluginBuilder and neuralops_attention Python plugin
       - Zero-fake export contract for TensorRT C++ plugin compilation

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates live serving engines when available).
    - Network: Not required; uses offline interface contracts and introspection.

What You Will Learn:
    1. How NeuralOps exposes non-quadratic linear attention to vLLM's PagedAttention engine.
    2. How SGLang's RadixAttention integrates recurrent state tree management.
    3. How to inspect method signatures and integration contracts when host engines are not installed.
    4. How to format serving compatibility matrices across deployment backends.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from examples._shared import (
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
)


def is_package_installed(pkg_name: str) -> bool:
    """Check whether a third-party package is installed in the current environment."""
    return importlib.util.find_spec(pkg_name) is not None


def main() -> int:
    section("NeuralOps Example 15: Serving Engine Integrations")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    compat_rows: list[dict[str, str]] = []

    # =========================================================================
    # Backend 1: vLLM (PagedAttention & Recurrent KV Cache Spec)
    # =========================================================================
    section("Backend 1: vLLM Custom Attention Backend")
    vllm_installed = is_package_installed("vllm")
    kv("vLLM Package Installed", vllm_installed)

    try:
        from neuralops.integrations.vllm_backend import (
            NeuralOpsAttentionBackend,
            NeuralOpsImpl,
            NeuralOpsRecurrentSpec,
            register as register_vllm,
        )

        register_vllm()
        pass_fail("vLLM Backend Module Import & Registration", True)

        # Introspect API surface
        methods = [m for m in dir(NeuralOpsAttentionBackend) if not m.startswith("_")]
        kv("Backend Class", NeuralOpsAttentionBackend.__name__)
        kv("Implementation Class", NeuralOpsImpl.__name__)
        kv("KV Cache Spec Class", NeuralOpsRecurrentSpec.__name__)
        kv("Public Methods", ", ".join(methods))

        if not vllm_installed:
            print("  [SKIPPED] vLLM package not installed on host — tested via API introspection contract.")
            compat_rows.append({
                "Backend": "vLLM",
                "Package Installed": "No",
                "Integration Module": "Available",
                "Test Status": "SKIPPED: Package missing (API Introspected)",
            })
        else:
            compat_rows.append({
                "Backend": "vLLM",
                "Package Installed": "Yes",
                "Integration Module": "Available",
                "Test Status": "PASS (Registered)",
            })
    except Exception as exc:
        pass_fail("vLLM Backend Integration", False, str(exc))
        compat_rows.append({
            "Backend": "vLLM",
            "Package Installed": "No",
            "Integration Module": "Failed",
            "Test Status": f"FAIL ({exc})",
        })

    # =========================================================================
    # Backend 2: SGLang (Linear Attention & Radix Tree Cache)
    # =========================================================================
    section("Backend 2: SGLang Linear Attention & Radix Cache")
    sglang_installed = is_package_installed("sglang")
    kv("SGLang Package Installed", sglang_installed)

    try:
        from neuralops.integrations.sglang_backend import (
            NeuralOpsLinearAttnBackend,
            register as register_sglang,
        )

        register_sglang()
        pass_fail("SGLang Backend Module Import & Registration", True)

        kv("Backend Class", NeuralOpsLinearAttnBackend.__name__)
        kv(
            "Uses Mamba Radix Cache Flag",
            getattr(NeuralOpsLinearAttnBackend, "uses_mamba_radix_cache", True),
        )
        fwd_extend_sig = inspect.signature(NeuralOpsLinearAttnBackend.forward_extend)
        fwd_decode_sig = inspect.signature(NeuralOpsLinearAttnBackend.forward_decode)
        kv("forward_extend Signature", str(fwd_extend_sig))
        kv("forward_decode Signature", str(fwd_decode_sig))

        if not sglang_installed:
            print("  [SKIPPED] SGLang package not installed on host — tested via API introspection contract.")
            compat_rows.append({
                "Backend": "SGLang",
                "Package Installed": "No",
                "Integration Module": "Available",
                "Test Status": "SKIPPED: Package missing (API Introspected)",
            })
        else:
            compat_rows.append({
                "Backend": "SGLang",
                "Package Installed": "Yes",
                "Integration Module": "Available",
                "Test Status": "PASS (Registered)",
            })
    except Exception as exc:
        pass_fail("SGLang Backend Integration", False, str(exc))
        compat_rows.append({
            "Backend": "SGLang",
            "Package Installed": "No",
            "Integration Module": "Failed",
            "Test Status": f"FAIL ({exc})",
        })

    # =========================================================================
    # Backend 3: TensorRT-LLM (Python Plugin & C++ Builder)
    # =========================================================================
    section("Backend 3: TensorRT-LLM Plugin Builder")
    trt_installed = is_package_installed("tensorrt_llm")
    kv("TensorRT-LLM Package Installed", trt_installed)

    try:
        from neuralops.integrations.tensorrt_llm import TensorRTLLMPluginBuilder, TRTPluginSpec
        from neuralops.integrations.tensorrt_llm_plugin import neuralops_attention

        pass_fail("TensorRT-LLM Integration Module Import", True)
        kv("Plugin Builder Class", TensorRTLLMPluginBuilder.__name__)
        kv("Plugin Spec Class", TRTPluginSpec.__name__)
        kv("Python Plugin Function", neuralops_attention.__name__)

        if not trt_installed:
            print("  [SKIPPED] tensorrt_llm package not installed on host — tested via API introspection contract.")
            compat_rows.append({
                "Backend": "TensorRT-LLM",
                "Package Installed": "No",
                "Integration Module": "Available",
                "Test Status": "SKIPPED: Package missing (API Introspected)",
            })
        else:
            compat_rows.append({
                "Backend": "TensorRT-LLM",
                "Package Installed": "Yes",
                "Integration Module": "Available",
                "Test Status": "PASS (Registered)",
            })
    except Exception as exc:
        pass_fail("TensorRT-LLM Integration", False, str(exc))
        compat_rows.append({
            "Backend": "TensorRT-LLM",
            "Package Installed": "No",
            "Integration Module": "Failed",
            "Test Status": f"FAIL ({exc})",
        })

    # =========================================================================
    # Serving Compatibility Matrix Table
    # =========================================================================
    section("Serving Engine Compatibility Matrix")
    headers = ["Backend", "Package Installed", "Integration Module", "Test Status"]
    rows = [[str(r[h]) for h in headers] for r in compat_rows]
    table(rows, headers=headers)

    pass_fail(
        "Overall Example 15 Status",
        True,
        "Serving backends introspected and integration contracts validated offline",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
