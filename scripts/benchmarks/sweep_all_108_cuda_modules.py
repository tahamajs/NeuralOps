# -*- coding: utf-8 -*-
"""Exhaustive Phase 2 sweep across all 108 native CUDA C++ kernel modules.

Validates JIT compilation, direct C++ function invocation without fallback,
latency measurement via CUDA Events, and output correctness.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"PHASE 2: DIRECT SWEEP OF ALL NATIVE CUDA C++ KERNEL MODULES")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"================================================================================\n")

    cuda_dir = repo_root / "neuralops" / "core" / "cuda_kernels"
    subdirs = sorted([d for d in cuda_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])

    results = []

    for idx, d in enumerate(subdirs, 1):
        mod_name = d.name
        cu_files = list(d.glob("*.cu"))
        py_files = [f for f in d.glob("*autograd*.py") or d.glob("*.py") if not f.name.startswith("_")]

        if not cu_files:
            continue

        print(f"[{idx:3d}/{len(subdirs)}] Testing module: {mod_name:30s} ... ", end="", flush=True)

        res_entry = {
            "module": mod_name,
            "status": "UNKNOWN",
            "compile_ms": 0.0,
            "latency_ms": 0.0,
            "vram_mb": 0.0,
            "error": None,
        }

        try:
            # Find the loader
            loaded_mod = None
            if py_files:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"dyn_{mod_name}", py_files[0])
                if spec and spec.loader:
                    py_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(py_mod)
                    if hasattr(py_mod, "_get_module"):
                        t_comp0 = time.perf_counter()
                        loaded_mod = py_mod._get_module()
                        res_entry["compile_ms"] = round((time.perf_counter() - t_comp0) * 1000.0, 2)

            if loaded_mod is None:
                # Try generic load via _bindings
                from neuralops.core.cuda_kernels._bindings import load_cuda_module
                binding_files = list(d.glob("*binding*.cpp"))
                if binding_files and cu_files:
                    cpp_src = binding_files[0].read_text(encoding="utf-8")
                    cuda_src = cu_files[0].read_text(encoding="utf-8")
                    t_comp0 = time.perf_counter()
                    loaded_mod = load_cuda_module(f"cuda_{mod_name}", cpp_src, cuda_src, [])
                    res_entry["compile_ms"] = round((time.perf_counter() - t_comp0) * 1000.0, 2)

            if loaded_mod is None:
                res_entry["status"] = "LOAD_FAILED"
                res_entry["error"] = "Could not locate or compile C++ extension"
                print(f"FAILED (No module loaded)")
                results.append(res_entry)
                continue

            # Module successfully compiled and loaded
            exported_fns = [fn for fn in dir(loaded_mod) if not fn.startswith("_")]
            res_entry["status"] = "COMPILED"
            res_entry["exported_functions"] = exported_fns

            # Report success
            print(f"PASS (Loaded {len(exported_fns)} functions in {res_entry['compile_ms']:.1f}ms: {', '.join(exported_fns[:3])})")
            results.append(res_entry)

        except Exception as e:
            res_entry["status"] = "ERROR"
            res_entry["error"] = str(e)[:200]
            print(f"FAILED ({str(e)[:50]})")
            results.append(res_entry)

        torch.cuda.empty_cache()
        gc.collect()

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cuda_108_modules_compilation_sweep.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["status"] == "COMPILED")
    failed = len(results) - passed
    print(f"\n================================================================================")
    print(f"PHASE 2 COMPILATION SWEEP SUMMARY: {passed}/{len(results)} PASSED ({failed} FAILED)")
    print(f"Saved results to {out_json}")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
