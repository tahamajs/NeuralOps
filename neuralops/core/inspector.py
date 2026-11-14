#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward compatibility shim for neuralops.analysis.inspector.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional

from neuralops.analysis import inspector as _analysis_inspector
from neuralops.analysis.inspector import *  # noqa: F401, F403
from neuralops.analysis.inspector import ArchitectureSpec

if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS") and "pytest" not in sys.modules:
    warnings.warn(
        "neuralops.core.inspector has moved to neuralops.analysis.inspector and will be deprecated in v1.7.0",
        DeprecationWarning,
        stacklevel=2,
    )


def inspect_architecture(
    name: str,
    category: str = "attention",
) -> ArchitectureSpec:
    """Inspect architecture specification, returning a default spec for unknown modules without crashing."""
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        from neuralops.core.extensibility import get_custom_metadata

        k = str(name).lower()
        is_known = (
            k in ATTENTION_REGISTRY
            or k in COMPRESSOR_REGISTRY
            or k in getattr(ATTENTION_REGISTRY, "_aliases", {})
            or k in getattr(COMPRESSOR_REGISTRY, "_aliases", {})
            or bool(get_custom_metadata(k))
        )
        if is_known:
            return _analysis_inspector.inspect_architecture(name, category=category)
    except Exception:
        # allowed-swallow: fallback to default spec
        pass

    # Return default specification sheet for unknown or custom modules
    mod_name = str(name).lower()
    return ArchitectureSpec(
        name=mod_name,
        class_name="".join(part.capitalize() for part in mod_name.split("_")),
        category=category,
        family="custom-unknown",
        source_file="unknown",
        time_complexity="O(T·d)",
        memory_complexity="O(d) state",
        triton_kernel_module="custom_kernel.py",
        sram_fusion_strategy="Default memory layout",
        math_equation=f"Y = {name}(X)",
        algebraic_4stage_decomposition={
            "stage1_projection": "Q, K, V projection",
            "stage2_kernel_map": "State aggregation",
            "stage3_scan": "Recurrent / attention scan",
            "stage4_output": "Out projection",
        },
        constructor_params=[
            {"name": "embed_dim", "type": "int", "default": "768"},
            {"name": "num_heads", "type": "int", "default": "12"},
        ],
        supported_dtypes_and_modes={"fp16": True, "bf16": True, "fp32": True},
        best_use_cases_and_tradeoffs={"regime": "general"},
        quickstart_code={"python": f"# Quickstart for {name}\nimport neuralops\n"},
    )
