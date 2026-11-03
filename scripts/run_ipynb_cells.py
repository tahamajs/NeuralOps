#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Robust headless runner for Jupyter Notebook code cells.
Handles IPython magics (%matplotlib, !shell) and provides standard notebook builtins.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import traceback

# Force UTF-8 stdout/stderr to prevent UnicodeEncodeError with terminal symbols/emojis
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Inject compatibility shim for torch.distributed._tensor on PyTorch 1.12
try:
    import torch.distributed._tensor
except (ImportError, ModuleNotFoundError):
    import types
    mod = types.ModuleType("torch.distributed._tensor")
    class DTensor: pass
    class Shard: pass
    class Replicate: pass
    mod.DTensor = DTensor
    mod.Shard = Shard
    mod.Replicate = Replicate
    mod.distribute_tensor = lambda t, *a, **kw: t
    mod.distribute_module = lambda m, *a, **kw: m
    sys.modules["torch.distributed._tensor"] = mod
    if "torch.distributed" in sys.modules:
        sys.modules["torch.distributed"]._tensor = mod

import torch


def clean_cell_source(source: str) -> str:
    """Preprocess code cell to strip or comment out IPython magics and shell escapes."""
    lines = source.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("%", "!")):
            cleaned.append(f"# [IPython Magic/Shell Stripped]: {line}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def execute_notebook(nb_path: str | Path) -> bool:
    nb_path = Path(nb_path)
    print(f"\n{'='*70}\n[START] Executing Notebook: {nb_path.name}\n{'='*70}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [c for c in nb_data.get("cells", []) if c.get("cell_type") == "code"]
    print(f"Total Code Cells to execute: {len(code_cells)}")

    # Setup execution namespace with common notebook builtins
    nb_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
        "display": print,
        "get_ipython": lambda: None,
    }

    # Pre-populate commonly expected modules in notebooks
    try:
        import numpy as np
        nb_globals["np"] = np
    except ImportError:
        pass

    try:
        import pandas as pd
        nb_globals["pd"] = pd
    except ImportError:
        pass

    passed = 0
    failed = 0

    for idx, cell in enumerate(code_cells):
        raw_src = "".join(cell.get("source", []))
        if not raw_src.strip():
            continue

        src = clean_cell_source(raw_src)
        print(f"\n--- Running Cell [{idx + 1}/{len(code_cells)}] ---")
        try:
            # First attempt
            exec(src, nb_globals)
            passed += 1
            print(f"Cell [{idx + 1}] PASSED")
        except TypeError as te:
            # Handle eval_strategy -> evaluation_strategy in older transformers
            if "eval_strategy" in str(te) and "eval_strategy" in src:
                try:
                    patched_src = src.replace("eval_strategy", "evaluation_strategy")
                    exec(patched_src, nb_globals)
                    passed += 1
                    print(f"Cell [{idx + 1}] PASSED (with eval_strategy compatibility patch)")
                    continue
                except Exception as inner_e:
                    failed += 1
                    print(f"Cell [{idx + 1}] FAILED with error: {type(inner_e).__name__}: {inner_e}")
            else:
                failed += 1
                print(f"Cell [{idx + 1}] FAILED with error: {type(te).__name__}: {te}")
        except Exception as e:
            failed += 1
            print(f"Cell [{idx + 1}] FAILED with error: {type(e).__name__}: {e}")

    print(f"\n{'='*70}\n[SUMMARY] Notebook: {nb_path.name} | Passed: {passed} | Failed: {failed}\n{'='*70}\n")
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ipynb_cells.py <notebook_path_or_--all>")
        sys.exit(1)

    if sys.argv[1] == "--all":
        nb_dir = REPO_ROOT / "notebooks"
        all_nbs = sorted(list(nb_dir.glob("*.ipynb")))
        print(f"\nDiscovered {len(all_nbs)} notebooks to execute in batch mode.\n")
        results = []
        for nb in all_nbs:
            try:
                ok = execute_notebook(nb)
                results.append((nb.name, "PASSED" if ok else "FAILED"))
            except Exception as e:
                print(f"[FATAL] Notebook {nb.name} crashed runner: {e}")
                results.append((nb.name, "CRASHED"))

        print("\n" + "=" * 70)
        print("FINAL BATCH NOTEBOOK EXECUTION SUMMARY:")
        print("=" * 70)
        for name, status in results:
            print(f"  {name:<60}: {status}")
        total_failed = sum(1 for _, s in results if s != "PASSED")
        print(f"\nTotal: {len(results)} | Passed: {len(results) - total_failed} | Failed/Crashed: {total_failed}\n")
        sys.exit(0 if total_failed == 0 else 1)
    else:
        success = True
        for arg in sys.argv[1:]:
            nb_path = Path(arg)
            ok = execute_notebook(nb_path)
            if not ok:
                success = False
        sys.exit(0 if success else 1)
