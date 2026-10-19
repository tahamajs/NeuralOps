#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
`doctor` command

diagnose installation.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def cmd_doctor() -> int:
    import neuralops
    ver = getattr(neuralops, "__version__", "1.4.2")

    rows = []
    all_ok = True

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_avail else 0
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        capability = torch.cuda.get_device_capability(0) if cuda_avail else (0, 0)
        rows.append(("PyTorch Version", torch.__version__, "pass", "Core runtime"))
        if cuda_avail:
            rows.append(("CUDA Device", f"{device_name} (CC {capability[0]}.{capability[1]}, {device_count} dev)", "pass", "GPU hardware"))
        else:
            rows.append(("CUDA Device", "Not Available (CPU only)", "warn", "GPU hardware"))
    except ImportError:
        rows.append(("PyTorch", "Not installed", "fail", "Core runtime"))
        all_ok = False

    try:
        import triton
        triton_cache = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache"))
        rows.append(("Triton JIT", f"v{triton.__version__} (Cache: {triton_cache})", "pass", "Kernel compiler"))
    except ImportError:
        rows.append(("Triton JIT", "Not installed (PyTorch fallbacks active)", "warn", "Kernel compiler"))

    try:
        repo_root = Path.cwd()
        git_lock = repo_root / ".git" / "index.lock"
        if git_lock.exists():
            rows.append(("Git Index Lock", f"Stale lock at {git_lock}", "warn", "Repository state"))
        else:
            rows.append(("Git Index Lock", "Clean", "pass", "Repository state"))

        total, used, free = shutil.disk_usage(repo_root)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        disk_status = "pass" if free_gb > 5.0 else "warn"
        rows.append(("Disk Space", f"{free_gb:.1f} GB free of {total_gb:.1f} GB", disk_status, "Filesystem"))
    except Exception as e:
        rows.append(("Disk / Git", f"Check error: {e}", "warn", "Filesystem"))

    try:
        import neuralops.attentions  # noqa: F401
        import neuralops.compression_engine  # noqa: F401
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        num_attn = len(ATTENTION_REGISTRY)
        num_comp = len(COMPRESSOR_REGISTRY)
        rows.append(("Attention Registry", f"{num_attn} modules loaded (Canonical: 131)", "pass", "Registry"))
        rows.append(("Compressor Registry", f"{num_comp} engines loaded (Canonical: 72)", "pass", "Registry"))
    except Exception as e:
        rows.append(("Registry Integrity", f"Failed to load: {e}", "fail", "Registry"))
        all_ok = False

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
    failed_sub = []
    for sub in subpackages:
        try:
            importlib.import_module(f"neuralops.{sub}")
        except Exception as e:
            failed_sub.append(f"{sub} ({e})")

    if not failed_sub:
        rows.append(("Subpackage Health", f"All {len(subpackages)} subpackages healthy", "pass", "Package integrity"))
    else:
        rows.append(("Subpackage Health", f"Errors in: {', '.join(failed_sub)}", "fail", "Package integrity"))
        all_ok = False

    if HAS_RICH and console is not None:
        table = Table(title=f"NeuralOps Production Diagnostics (v{ver})", header_style="bold cyan")
        table.add_column("Status", justify="center", width=8)
        table.add_column("Component", style="bold", width=22)
        table.add_column("Category", style="dim", width=18)
        table.add_column("Details", min_width=35)

        for comp, details, status, cat in rows:
            if status == "pass":
                stat_str = "[bold green]✓ OK[/]"
            elif status == "warn":
                stat_str = "[bold yellow]⚠ WARN[/]"
            else:
                stat_str = "[bold red]✗ FAIL[/]"
            table.add_row(stat_str, comp, cat, details)

        console.print(table)
        if all_ok:
            console.print("[bold green]✓ Production environment diagnostics passed.[/]\n")
        else:
            console.print("[bold red]✗ One or more diagnostics failed. Please resolve above issues.[/]\n")
    else:
        print("=" * 65)
        print(f"NeuralOps Production Diagnostics (v{ver})")
        print("=" * 65)
        for comp, details, status, cat in rows:
            icon = "✓ OK  " if status == "pass" else ("⚠ WARN" if status == "warn" else "✗ FAIL")
            print(f"[{icon}] {comp:22s} ({cat:16s}): {details}")
        print("=" * 65)
        if all_ok:
            print("✓ Production environment diagnostics passed.\n")
        else:
            print("✗ One or more diagnostics failed.\n")

    return 0 if all_ok else 1
