#!/usr/bin/env python3
"""
Kaggle GPU/TPU Pipeline Launcher for Efficient Attention Research
===================================================================
Automated environment setup, dependency verification, hardware acceleration
configuration, and pipeline execution for Kaggle Notebooks & Kernels.

Usage in Kaggle Notebook Cell
-----------------------------
!python run_kaggle.py train --attn_type full --task sst2 --epochs 3
!python run_kaggle.py sweep --attn_types full nha lsh --seeds 42 123 --task sst2
!python run_kaggle.py list
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_kaggle_environment():
    """Detect Kaggle environment, install missing packages, and setup paths."""
    print("=" * 60)
    print("🚀 Initializing Kaggle Pipeline Launcher")
    print("=" * 60)

    is_kaggle = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    if is_kaggle:
        print("✓ Kaggle environment detected (/kaggle/working)")
    else:
        print("ℹ Standard environment detected")

    # Required packages check
    reqs = ["datasets", "transformers", "accelerate", "evaluate", "scipy", "sklearn", "deepdiff", "matplotlib"]
    missing = []
    for req in reqs:
        try:
            __import__(req if req != "sklearn" else "sklearn")
        except ImportError:
            missing.append(req)

    if missing:
        print(f"📦 Installing missing dependencies: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("✓ Dependencies installed successfully")
    else:
        print("✓ All dependencies verified")

    # Add framework to sys.path
    here = Path(__file__).resolve().parent
    codes = here.parent
    if str(codes) not in sys.path:
        sys.path.insert(0, str(codes))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # GPU / Hardware detection
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"⚡ GPU Acceleration Enabled: {device_name} ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        print("⚡ Apple Silicon (MPS) Acceleration Enabled")
    else:
        print("⚠️ No GPU detected. Running on CPU.")

    print("=" * 60)


if __name__ == "__main__":
    setup_kaggle_environment()

    if "/kaggle/working" in os.getcwd() or os.path.exists("/kaggle/working"):
        if len(sys.argv) > 1 and sys.argv[1] not in ("list", "-h", "--help"):
            if "--output_dir" not in sys.argv:
                sys.argv.extend(["--output_dir", "/kaggle/working/results"])
            if "--log_dir" not in sys.argv:
                sys.argv.extend(["--log_dir", "/kaggle/working/logs"])

    import framework.main as pipeline_main
    pipeline_main.main()
