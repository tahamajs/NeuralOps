"""Plan subcommand for CLI v2."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional


def setup_plan_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for plan subcommand."""
    parser = subparsers.add_parser("plan", help="Dry-run and estimate execution grid")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML benchmark configuration file")
    parser.add_argument("--modules", type=str, default="all", help="Modules to benchmark")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device (cpu, cuda, etc.)")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Target model")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated target models")
    parser.add_argument("--all-models", action="store_true", help="Benchmark all supported models")
    parser.add_argument("--dataset", type=str, default="sst2", help="Target dataset")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated target datasets")
    parser.add_argument("--all-datasets", action="store_true", help="Benchmark all supported datasets")
    parser.add_argument("--rank", type=int, default=64, help="Rank for low-rank attention/compressor")
    parser.add_argument("--ranks", type=str, default=None, help="Comma-separated ranks")
    parser.add_argument("--yes", action="store_true", help="Confirm execution without prompt")
    parser.add_argument("--json", action="store_true", help="Output plan details as JSON")
    return parser


def handle_plan(args: argparse.Namespace) -> int:
    """Execute plan dry-run estimation."""
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Estimate plan grid
    modules_str = getattr(args, "modules", "all") or "all"
    if modules_str == "all":
        mod_count = 1
    else:
        mod_count = len(modules_str.split(","))

    models_count = (
        10
        if getattr(args, "all_models", False)
        else (len(args.models.split(",")) if getattr(args, "models", None) else 1)
    )
    datasets_count = (
        22
        if getattr(args, "all_datasets", False)
        else (len(args.datasets.split(",")) if getattr(args, "datasets", None) else 1)
    )
    ranks_count = len(args.ranks.split(",")) if getattr(args, "ranks", None) else 1

    total_runs = mod_count * models_count * datasets_count * ranks_count
    est_seconds = total_runs * 180
    est_hours = est_seconds / 3600.0

    if getattr(args, "json", False):
        import json

        plan_data = {
            "modules": mod_count,
            "target_models": models_count,
            "datasets": datasets_count,
            "ranks": ranks_count,
            "total_runs": total_runs,
            "estimated_seconds": est_seconds,
            "estimated_hours": est_hours,
        }
        print(json.dumps(plan_data, indent=2))
        return 0

    print("═════════════════════════════════════════════════════")
    print("Benchmark Plan Summary (v2):")
    print(f"  Modules:  {mod_count}")
    print(f"  Target Models:  {models_count}")
    print(f"  Datasets:       {datasets_count}")
    print(f"  Ranks:    {ranks_count}")
    print(f"  Total planned runs: {total_runs}")
    print(f"  Estimated duration: ~{est_hours:.1f} hours")
    print("═════════════════════════════════════════════════════")
    return 0
