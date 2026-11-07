#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Unified CLI Main Router.

Central entrypoint dispatching commands across benchmarks, compression,
training, evaluation, surgery, and system diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from neuralops import __version__
from neuralops.cli.bench_cmd import add_benchmark_parser


def build_main_parser() -> argparse.ArgumentParser:
    """Construct the unified top-level NeuralOps CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="NeuralOps Unified CLI & Benchmark Orchestrator.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"NeuralOps v{__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Command family or operation",
    )

    # 1. Benchmark router (Phase 1, 2, 3)
    add_benchmark_parser(subparsers)

    # 2. Additional core CLI subcommands
    try:
        from neuralops.cli.compress_cmd import build_compress_parser, cmd_compress

        compress_p = build_compress_parser(subparsers.add_parser("compress", help="Model compression"))
        compress_p.set_defaults(handler=cmd_compress)
    except Exception:
        pass

    try:
        from neuralops.cli.eval_cmd import build_eval_parser, cmd_eval

        eval_p = build_eval_parser(subparsers.add_parser("eval", aliases=["evaluate"], help="Model evaluation"))
        eval_p.set_defaults(handler=cmd_eval)
    except Exception:
        pass

    try:
        from neuralops.cli.list_cmd import build_list_parser, cmd_list

        list_p = build_list_parser(subparsers.add_parser("list", help="List registered components"))
        list_p.set_defaults(handler=cmd_list)
    except Exception:
        pass

    try:
        from neuralops.cli.serve_cmd import build_serve_parser, cmd_serve

        serve_p = build_serve_parser(subparsers.add_parser("serve", help="Launch inference serving server"))
        serve_p.set_defaults(handler=cmd_serve)
    except Exception:
        pass

    try:
        from neuralops.cli.test_cmd import build_test_parser, cmd_test

        test_p = build_test_parser(subparsers.add_parser("test", help="Run diagnostic health tests"))
        test_p.set_defaults(handler=cmd_test)
    except Exception:
        pass

    try:
        from neuralops.cli.train_cmd import build_train_parser, cmd_train

        train_p = build_train_parser(subparsers.add_parser("train", help="Pretrain or fine-tune models"))
        train_p.set_defaults(handler=cmd_train)
    except Exception:
        pass

    try:
        from neuralops.cli.doctor import cmd_doctor

        doctor_p = subparsers.add_parser("doctor", help="Inspect local hardware, toolchains, and environment health")
        doctor_p.add_argument("--json", action="store_true", default=False, help="Output diagnostics as JSON")
        doctor_p.set_defaults(handler=cmd_doctor)
    except Exception:
        pass

    try:
        from neuralops.cli.info import cmd_info

        info_p = subparsers.add_parser("info", help="Display hardware, framework, and registered component status")
        info_p.add_argument("--json", action="store_true", default=False, help="Output system information as JSON")
        info_p.set_defaults(handler=cmd_info)
    except Exception:
        pass

    try:
        from neuralops.cli.surgery_cmd import build_surgery_parser, build_transplant_parser, cmd_surgery, cmd_transplant

        surg_p = build_surgery_parser(
            subparsers.add_parser(
                "surgery",
                aliases=["recipe"],
                help="Plan, validate, and execute declarative transformation recipes",
            )
        )
        surg_p.set_defaults(handler=cmd_surgery)

        trans_p = build_transplant_parser(
            subparsers.add_parser(
                "transplant",
                help="Surgically transplant attention mechanisms into model checkpoints",
            )
        )
        trans_p.set_defaults(handler=cmd_transplant)
    except Exception:
        pass

    try:
        from neuralops.ui.cli import build_ui_parser, cmd_ui

        ui_p = build_ui_parser(
            subparsers.add_parser(
                "ui",
                help="Launch NeuralOps Web UI",
            )
        )
        ui_p.set_defaults(handler=cmd_ui)
    except Exception:
        pass

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main execution router for NeuralOps CLI."""
    parser = build_main_parser()
    raw_args = sys.argv[1:] if argv is None else list(argv)

    if not raw_args or raw_args in (["-h"], ["--help"]):
        parser.print_help()
        return 0

    args = parser.parse_args(raw_args)

    # Dispatch to attached handler
    if hasattr(args, "func") and args.func is not None:
        return args.func(args)
    if hasattr(args, "benchmark_handler") and args.benchmark_handler is not None:
        return args.benchmark_handler(args)
    if hasattr(args, "handler") and args.handler is not None:
        return args.handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
