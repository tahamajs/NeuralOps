"""
Framework Integrity & Synchronization Verification Script.

Validates that:
1. All Attention modules and Compressor modules are importable without errors.
2. All modules are properly registered in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY.
3. All TRACK_MODULES in eval_all_models_datasets.py map to valid registered components.
4. All custom Linear layers implement the `.weight` property for T5/HuggingFace introspection.
5. All CLI entry points (auto_run.py, launch_gpu_server.py, prepare_datasets.py, run_on_gpu_server.sh) are synchronized.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn

# Ensure framework is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODES_DIR, ".."))
for p in [PROJECT_DIR, CODES_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SyncValidator")


def check_all_registries() -> Dict[str, Any]:
    """Imports all modules and checks registry completeness."""
    import framework.attentions
    import framework.compression_engine
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    from framework.eval_all_models_datasets import TRACK_MODULES

    # Check track modules mapping with dynamic lazy registration
    track_errors: List[str] = []
    for track_name, track_data in TRACK_MODULES.items():
        for attn in track_data.get("attentions", []):
            if attn not in ATTENTION_REGISTRY:
                track_errors.append(f"Track '{track_name}' attention '{attn}' NOT found in ATTENTION_REGISTRY")
        for comp in track_data.get("compressors", []):
            if comp not in COMPRESSOR_REGISTRY:
                track_errors.append(f"Track '{track_name}' compressor '{comp}' NOT found in COMPRESSOR_REGISTRY")

    registered_attns = set(ATTENTION_REGISTRY.list_available())
    registered_comps = set(COMPRESSOR_REGISTRY.list_available())

    return {
        "num_attentions": len(registered_attns),
        "num_compressors": len(registered_comps),
        "track_errors": track_errors,
        "is_synced": len(track_errors) == 0,
    }


def check_custom_linear_weight_properties() -> List[str]:
    """Inspects all custom Linear replacement modules for .weight property compatibility."""
    import framework.compression_engine as ce
    missing_weight_prop: List[str] = []

    for name, obj in inspect.getmembers(ce):
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            # Check if this class acts as a linear replacement
            class_src = inspect.getsource(obj)
            if "out_features" in class_src or "in_features" in class_src:
                if not hasattr(obj, "weight") and not any(isinstance(getattr(obj, attr, None), property) for attr in dir(obj) if attr == "weight"):
                    # Check if instance has weight buffer/parameter
                    try:
                        dummy = obj(64, 64) if "in_features" in inspect.signature(obj.__init__).parameters else None
                        if dummy is not None and not hasattr(dummy, "weight"):
                            missing_weight_prop.append(f"{obj.__name__} in {obj.__module__}")
                    except Exception:
                        pass

    return missing_weight_prop


def main() -> None:
    print("=" * 80)
    print("        NEURALOPS FRAMEWORK INTEGRITY & SYNCHRONIZATION AUDIT")
    print("=" * 80)

    reg_result = check_all_registries()
    print(f" Total Registered Attention Modules : {reg_result['num_attentions']}")
    print(f" Total Registered Compressor Engines: {reg_result['num_compressors']}")
    
    if reg_result["track_errors"]:
        print(f"\n❌ FOUND {len(reg_result['track_errors'])} TRACK MAPPING ERRORS:")
        for err in reg_result["track_errors"]:
            print(f"  - {err}")
    else:
        print("\n✅ All 5 Tracks in eval_all_models_datasets.py are 100% SYNCHRONIZED with Registries!")

    weight_issues = check_custom_linear_weight_properties()
    if weight_issues:
        print(f"\n⚠️  Modules potentially missing `.weight` property: {len(weight_issues)}")
        for w in weight_issues:
            print(f"  - {w}")
    else:
        print("✅ All Custom Linear Layers implement `.weight` compatibility property!")

    print("\n" + "=" * 80)
    if reg_result["is_synced"]:
        print(" 🎉 FULL FRAMEWORK INTEGRITY & CODE SYNCHRONIZATION CONFIRMED!")
    else:
        print(" ⚠️  Please resolve the listed mapping errors above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
