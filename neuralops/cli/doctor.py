"""`doctor` command — diagnose installation."""
import importlib


def cmd_doctor():
    print("Running diagnostics...")
    print()

    # Check torch
    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
    except ImportError:
        print("[FAIL] torch not installed")

    # Check triton
    try:
        import triton
        print(f"[OK] triton {triton.__version__}")
    except ImportError:
        print("[WARN] triton not installed (CPU fallback will be used)")

    # Check registry
    try:
        from neuralops.core.registry import (
            ATTENTION_REGISTRY, COMPRESSOR_REGISTRY,
        )
        import neuralops.attentions  # noqa
        import neuralops.compression_engine  # noqa
        print(f"[OK] Attention registry: {len(ATTENTION_REGISTRY.keys())}")
        print(f"[OK] Compressor registry: {len(COMPRESSOR_REGISTRY.keys())}")
    except Exception as e:
        print(f"[FAIL] Registry: {e}")

    # Check subpackages
    for sub in ["core", "attentions", "compression_engine",
                "benchmarks", "trainer", "evaluation", "data", "utils"]:
        try:
            importlib.import_module(f"neuralops.{sub}")
            print(f"[OK] neuralops.{sub}")
        except Exception as e:
            print(f"[FAIL] neuralops.{sub}: {e}")

    print()
    print("Diagnostics complete.")
    return 0
