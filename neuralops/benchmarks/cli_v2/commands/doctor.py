"""Doctor subcommand for CLI v2."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional


def setup_doctor_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for doctor subcommand."""
    parser = subparsers.add_parser("doctor", help="Run comprehensive diagnostics on environment")
    parser.add_argument("--json", action="store_true", help="Format output as JSON")
    parser.add_argument("--device", type=str, default=None, help="Device setting (cpu, cuda, etc.)")
    return parser


def handle_doctor(args: argparse.Namespace) -> int:
    """Execute doctor environment diagnostics."""
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from neuralops.benchmarks.doctor import cmd_doctor

    return cmd_doctor()
