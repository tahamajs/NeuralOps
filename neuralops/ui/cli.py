#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Web UI Command Module.

Implements argparse parser configuration and command handler for the NeuralOps Web UI.
"""

from __future__ import annotations

import argparse
from typing import Optional


def _add_ui_arguments(parser: argparse.ArgumentParser) -> None:
    """Add standard UI runtime arguments to parser."""
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Cloudflare tunnel to share the UI",
    )
    parser.add_argument(
        "--static",
        type=str,
        default=None,
        metavar="DIR",
        help="Generate a static HTML bundle for HuggingFace Spaces deployment",
    )
    parser.add_argument(
        "--read-write",
        action="store_true",
        dest="read_write",
        default=False,
        help="Enable read-write operations (benchmarking/distillation execution)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        default=False,
        help="Do not automatically launch web browser on startup",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        metavar="USER:PASS",
        help="Basic authentication credentials for public access (required with --share)",
    )


def build_ui_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops ui subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops ui",
        description="Launch NeuralOps publication-grade Web UI dashboard.",
    )
    _add_ui_arguments(parser)

    # Support optional `start` sub-action for parity with start workflows
    ui_subparsers = parser.add_subparsers(dest="ui_action", help="UI sub-actions")
    start_p = ui_subparsers.add_parser("start", help="Start the Web UI server")
    _add_ui_arguments(start_p)
    start_p.set_defaults(handler=cmd_ui)

    return parser


def cmd_ui(args: argparse.Namespace) -> int:
    """Execute the UI CLI command (Phase 1 scaffold)."""
    port = getattr(args, "port", 8080)
    host = getattr(args, "host", "127.0.0.1")
    share = getattr(args, "share", False)
    static_path = getattr(args, "static", None)
    read_write = getattr(args, "read_write", False)
    no_browser = getattr(args, "no_browser", False)
    auth = getattr(args, "auth", None)

    print(
        f"NeuralOps Web UI (Phase 1 Scaffold): "
        f"host={host}, port={port}, share={share}, static={static_path}, "
        f"read_write={read_write}, no_browser={no_browser}, auth={auth}"
    )
    print("UI command not yet implemented — Phase 2 will add the server.")
    return 0


__all__ = ["build_ui_parser", "cmd_ui"]
