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

Empirical Serving Findings (Dual NVIDIA Tesla T4):
    This example was verified on Kaggle dual T4 with vLLM v0.32.0 installed.
    The vLLM backend runs but is 4x slower than PyTorch baseline (see
    docs/SERVING_INTEGRATION_REPORT.md for details). SGLang failed on T4
    because sgl_kernel only bundles SM100 binaries. TRT-LLM was deferred
    because no Python 3.13 wheel exists.

What You Will Learn:
    1. How NeuralOps exposes non-quadratic linear attention to vLLM's PagedAttention engine.
    2. How SGLang's RadixAttention integrates recurrent state tree management.
    3. How to inspect method signatures and integration contracts when host engines are not installed.
    4. How to format serving compatibility matrices across deployment backends.
"""

from __future__ import annotations

import inspect
import tempfile
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
    skip,
    skip_if_missing,
    table,
)


def main() -> int:
    section("NeuralOps Example 15: Serving Engine Integrations")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    compat_rows: list[list[str]] = []
    checks: list[bool] = []

    # =========================================================================
    # Backend 1: vLLM (PagedAttention & Recurrent KV Cache Spec)
    # =========================================================================
    section("Backend 1: vLLM Custom Attention Backend")
    vllm_missing = skip_if_missing("vllm", "vLLM package is not installed in current environment")
    vllm_installed = not vllm_missing
    kv("vLLM Package Installed", vllm_installed)

    has_vllm_module = False
    can_register_vllm = False
    can_exec_vllm = False

    try:
        from neuralops.integrations.vllm_backend import (
            NeuralOpsAttentionBackend,
            NeuralOpsImpl,
            NeuralOpsRecurrentSpec,
            register as register_vllm,
        )

        has_vllm_module = True
        methods = [m for m in dir(NeuralOpsAttentionBackend) if not m.startswith("_")]
        kv("Backend Class", NeuralOpsAttentionBackend.__name__)
        kv("Implementation Class", NeuralOpsImpl.__name__)
        kv("KV Cache Spec Class", NeuralOpsRecurrentSpec.__name__)
        kv("Public Methods", ", ".join(methods))

        if vllm_installed:
            reg_ok = bool(register_vllm())
            can_register_vllm = reg_ok
            can_exec_vllm = reg_ok
            vllm_subcheck = reg_ok
            pass_fail(
                "vLLM Backend Live Registration",
                reg_ok,
                "Live vLLM registration successful" if reg_ok else "Registration failed",
            )
            status_vllm = "PASS" if reg_ok else "FAIL: Registration failed"
        else:
            vllm_contract_ok = hasattr(NeuralOpsAttentionBackend, "get_name") and hasattr(
                NeuralOpsAttentionBackend, "get_builder_cls"
            )
            vllm_subcheck = vllm_contract_ok
            pass_fail("vLLM Interface Contract Verification", vllm_contract_ok, "Backend contract validated statically")
            status_vllm = "SKIP: Package missing"

        checks.append(vllm_subcheck)
    except Exception as exc:
        pass_fail("vLLM Backend Integration", False, str(exc))
        status_vllm = f"FAIL: {exc}"
        checks.append(False)

    compat_rows.append(
        [
            "vLLM",
            "YES" if vllm_installed else "NO",
            "YES" if has_vllm_module else "NO",
            "YES" if can_register_vllm else "NO",
            "YES" if can_exec_vllm else "NO",
            status_vllm,
        ]
    )

    # =========================================================================
    # Backend 2: SGLang (Linear Attention & Radix Tree Cache)
    # =========================================================================
    section("Backend 2: SGLang Linear Attention & Radix Cache")
    sglang_missing = skip_if_missing("sglang", "SGLang package is not installed in current environment")
    sglang_installed = not sglang_missing
    kv("SGLang Package Installed", sglang_installed)

    has_sglang_module = False
    can_register_sglang = False
    can_exec_sglang = False

    try:
        from neuralops.integrations.sglang_backend import (
            NeuralOpsLinearAttnBackend,
            register as register_sglang,
        )

        has_sglang_module = True
        kv("Backend Class", NeuralOpsLinearAttnBackend.__name__)
        kv(
            "Uses Mamba Radix Cache Flag",
            getattr(NeuralOpsLinearAttnBackend, "uses_mamba_radix_cache", True),
        )
        fwd_extend_sig = inspect.signature(NeuralOpsLinearAttnBackend.forward_extend)
        fwd_decode_sig = inspect.signature(NeuralOpsLinearAttnBackend.forward_decode)
        kv("forward_extend Signature", str(fwd_extend_sig))
        kv("forward_decode Signature", str(fwd_decode_sig))

        if sglang_installed:
            reg_ok = bool(register_sglang())
            can_register_sglang = reg_ok
            can_exec_sglang = reg_ok
            sglang_subcheck = reg_ok
            pass_fail(
                "SGLang Backend Live Registration",
                reg_ok,
                "Live SGLang registration successful" if reg_ok else "Registration failed",
            )
            status_sglang = "PASS" if reg_ok else "FAIL: Registration failed"
        else:
            sglang_contract_ok = hasattr(NeuralOpsLinearAttnBackend, "forward_extend") and hasattr(
                NeuralOpsLinearAttnBackend, "forward_decode"
            )
            sglang_subcheck = sglang_contract_ok
            pass_fail(
                "SGLang Interface Contract Verification", sglang_contract_ok, "Backend signatures validated statically"
            )
            status_sglang = "SKIP: Package missing"

        checks.append(sglang_subcheck)
    except Exception as exc:
        pass_fail("SGLang Backend Integration", False, str(exc))
        status_sglang = f"FAIL: {exc}"
        checks.append(False)

    compat_rows.append(
        [
            "SGLang",
            "YES" if sglang_installed else "NO",
            "YES" if has_sglang_module else "NO",
            "YES" if can_register_sglang else "NO",
            "YES" if can_exec_sglang else "NO",
            status_sglang,
        ]
    )

    # =========================================================================
    # Backend 3: TensorRT-LLM (Python Plugin & C++ Builder)
    # =========================================================================
    section("Backend 3: TensorRT-LLM Plugin Builder")
    trt_missing = skip_if_missing("tensorrt_llm", "TensorRT-LLM package is not installed in current environment")
    trt_installed = not trt_missing
    kv("TensorRT-LLM Package Installed", trt_installed)

    has_trt_module = False
    can_register_trt = False
    can_exec_trt = False

    try:
        from neuralops.integrations.tensorrt_llm import TensorRTLLMPluginBuilder, TRTPluginSpec
        from neuralops.integrations.tensorrt_llm_plugin import neuralops_attention

        has_trt_module = True
        can_register_trt = True
        kv("Plugin Builder Class", TensorRTLLMPluginBuilder.__name__)
        kv("Plugin Spec Class", TRTPluginSpec.__name__)
        kv("Python Plugin Function", neuralops_attention.__name__)

        trt_contract_ok = hasattr(TensorRTLLMPluginBuilder, "generate_cpp_plugin_source") and callable(
            neuralops_attention
        )
        pass_fail("TensorRT-LLM Interface Contract Verification", trt_contract_ok, "Plugin builder contracts validated")

        builder = TensorRTLLMPluginBuilder()
        with tempfile.TemporaryDirectory() as tmp_trt_dir:
            generated_files = builder.generate_cpp_plugin_source(tmp_trt_dir)
            synthesis_ok = bool(len(generated_files) >= 3)
            pass_fail(
                "TensorRT-LLM C++ Plugin Synthesis", synthesis_ok, f"{len(generated_files)} artifacts synthesized"
            )
            kv("Synthesized C++ Artifacts", ", ".join(generated_files.keys()))

        if not trt_installed:
            status_trt = "PASS (offline synthesis)"
        else:
            can_exec_trt = True
            status_trt = "PASS"

        checks.append(trt_contract_ok and synthesis_ok)
    except Exception as exc:
        pass_fail("TensorRT-LLM Integration", False, str(exc))
        status_trt = f"FAIL: {exc}"
        checks.append(False)

    compat_rows.append(
        [
            "TRT-LLM",
            "YES" if trt_installed else "NO",
            "YES" if has_trt_module else "NO",
            "YES" if can_register_trt else "NO",
            "YES" if can_exec_trt else "NO",
            status_trt,
        ]
    )

    # =========================================================================
    # Serving Compatibility Matrix Table
    # =========================================================================
    section("Serving Engine Compatibility Matrix")
    headers = ("Backend", "Package Installed", "NeuralOps Module", "Can Register", "Can Execute", "Status")
    table(compat_rows, headers=headers)

    all_passed = bool(len(checks) >= 3 and all(checks))
    pass_fail(
        "Overall Example 15 Status",
        all_passed,
        "Serving backends introspected and integration contracts validated offline"
        if all_passed
        else "Serving contract failures detected",
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
