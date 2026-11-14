"""UI subcommand for CLI v2."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional


def setup_ui_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for ui subcommand."""
    parser = subparsers.add_parser("ui", help="Launch NeuralOps Web UI dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server bind host (default: 127.0.0.1)")
    parser.add_argument("--device", type=str, default=None, help="Device setting (cpu, cuda, etc.)")
    return parser


def handle_ui(args: argparse.Namespace) -> int:
    """Execute UI startup or configuration."""
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from neuralops.ui.cli import cmd_ui

    return cmd_ui(args)
