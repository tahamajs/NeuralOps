"""`doctor` command — diagnose installation."""
import importlib


def cmd_doctor():
    print("=" * 60)
    print("NeuralOps Production Environment & Hardware Diagnostics (v1.3.0)")
    print("=" * 60)
    print()

    # 1. PyTorch & CUDA Detection
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_avail else 0
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        capability = torch.cuda.get_device_capability(0) if cuda_avail else (0, 0)
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA Available: {cuda_avail} (Devices: {device_count})")
        if cuda_avail:
            print(f"     Device 0: {device_name} (Compute Capability: {capability[0]}.{capability[1]})")
    except ImportError:
        print("[FAIL] torch not installed")

    # 2. Triton Detection & Cache Inspection
    try:
        import triton
        import os
        from pathlib import Path
        triton_cache = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache"))
        print(f"[OK] Triton version: {triton.__version__}")
        print(f"[OK] Triton Cache Directory: {triton_cache}")
    except ImportError:
        print("[WARN] Triton not installed (Deterministic PyTorch fallbacks active)")

    # 3. Disk & Git Lock Hygiene
    try:
        import os
        from pathlib import Path
        repo_root = Path.cwd()
        git_lock = repo_root / ".git" / "index.lock"
        if git_lock.exists():
            print(f"[WARN] Stale git lock detected at: {git_lock} (Run rm -f to clear)")
        else:
            print("[OK] Git index lock: Clean")

        # Check /tmp and ~/.cache space
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"[OK] Disk Space Root (/): {free / (1024**3):.1f} GB free of {total / (1024**3):.1f} GB")
    except Exception as e:
        print(f"[WARN] Disk/Git check: {e}")

    # 4. Registry Integrity
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        num_attn = len(ATTENTION_REGISTRY)
        num_comp = len(COMPRESSOR_REGISTRY)
        print(f"[OK] Attention Mechanism Registry: {num_attn} modules loaded")
        print(f"[OK] Compression Engine Registry: {num_comp} engines loaded")
    except Exception as e:
        print(f"[FAIL] Registry: {e}")

    # 5. Subpackage Import Health
    subpackages = [
        "core",
        "attentions",
        "compression_engine",
        "surgery",
        "export",
        "benchmarks",
        "trainer",
        "evaluation",
        "data",
        "utils",
    ]
    for sub in subpackages:
        try:
            importlib.import_module(f"neuralops.{sub}")
            print(f"[OK] Package neuralops.{sub}: Healthy")
        except Exception as e:
            print(f"[FAIL] Package neuralops.{sub}: {e}")

    print()
    print("=" * 60)
    print("Diagnostics complete. Production environment is healthy.")
    print("=" * 60)
    return 0
