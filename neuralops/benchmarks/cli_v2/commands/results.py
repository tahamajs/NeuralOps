"""Results subcommand for CLI v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

    fmt = getattr(args, "format", "table")
    dir_path = Path(out_dir)

    records: List[Dict[str, Any]] = []
    if dir_path.is_dir():
        for file in sorted(dir_path.glob("*.json")):
            try:
                content = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(content, dict):
                    records.append(content)
            except Exception:  # allowed-swallow: corrupted or non-json files ignored during scanning
                pass

    if fmt == "json":
        print(json.dumps({"results_dir": str(out_dir), "record_count": len(records), "records": records}, indent=2))
    elif getattr(args, "list", False):
        print(f"Found {len(records)} benchmark records in {out_dir}")
        for r in records[:10]:
            print(f"  - {r.get('name', 'unnamed')}: status={r.get('status', 'unknown')}")

    return 0
