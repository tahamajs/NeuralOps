"""Results subcommand for CLI v2."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional


def setup_results_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for results subcommand."""
    parser = subparsers.add_parser("results", help="Inspect and query benchmark results")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory to inspect")
    parser.add_argument(
        "--format", type=str, choices=["table", "json", "csv", "markdown"], default="table", help="Output format"
    )
    parser.add_argument("--latest", action="store_true", help="Display latest benchmark run record")
    parser.add_argument("--list", action="store_true", help="List benchmark records")
    parser.add_argument("--module", type=str, default=None, help="Filter by module name")
    parser.add_argument("--device", type=str, default=None, help="Device setting (cpu, cuda, etc.)")
    return parser


def handle_results(args: argparse.Namespace) -> int:
    """Execute results inspection."""
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_dir = getattr(args, "output_dir", "results")
    print(f"Results output directory: {out_dir}")
    return 0
