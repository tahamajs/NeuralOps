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
import logging
import sys
from typing import Optional, Sequence

from neuralops import __version__

logger = logging.getLogger("NeuralOps.CLI.Main")


def build_main_parser() -> argparse.ArgumentParser:
    """Construct the unified top-level NeuralOps CLI argument parser.

    All subcommands are registered fail-fast: any broken module import raises
    immediately with a full traceback and module identity, never silently dropping commands.
    Every subcommand handler adheres to the standard `func` dispatch contract via `set_defaults(func=...)`.
    """
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
    logger.debug("Registering subcommand 'benchmark' from 'neuralops.cli.bench_cmd'")
    from neuralops.cli.bench_cmd import add_benchmark_parser

    add_benchmark_parser(subparsers)

    # 2. Additional core CLI subcommands
    logger.debug("Registering subcommand 'compress' from 'neuralops.cli.compress_cmd'")
    from neuralops.cli.compress_cmd import build_compress_parser, cmd_compress

    compress_p = build_compress_parser(subparsers.add_parser("compress", help="Model compression"))
    compress_p.set_defaults(func=cmd_compress)

    logger.debug("Registering subcommand 'eval' from 'neuralops.cli.eval_cmd'")
    from neuralops.cli.eval_cmd import build_eval_parser, cmd_eval

    eval_p = build_eval_parser(subparsers.add_parser("eval", aliases=["evaluate"], help="Model evaluation"))
    eval_p.set_defaults(func=cmd_eval)

    logger.debug("Registering subcommand 'list' from 'neuralops.cli.list_cmd'")
    from neuralops.cli.list_cmd import build_list_parser, cmd_list

    list_p = build_list_parser(subparsers.add_parser("list", help="List registered components"))
    list_p.set_defaults(func=cmd_list)

    logger.debug("Registering subcommand 'serve' from 'neuralops.cli.serve_cmd'")
    from neuralops.cli.serve_cmd import build_serve_parser, cmd_serve

    serve_p = build_serve_parser(subparsers.add_parser("serve", help="Launch inference serving server"))
    serve_p.set_defaults(func=cmd_serve)

    logger.debug("Registering subcommand 'test' from 'neuralops.cli.test_cmd'")
    from neuralops.cli.test_cmd import build_test_parser, cmd_test

    test_p = build_test_parser(subparsers.add_parser("test", help="Run diagnostic health tests"))
    test_p.set_defaults(func=cmd_test)

    logger.debug("Registering subcommand 'train' from 'neuralops.cli.train_cmd'")
    from neuralops.cli.train_cmd import build_train_parser, cmd_train

    train_p = build_train_parser(subparsers.add_parser("train", help="Pretrain or fine-tune models"))
    train_p.set_defaults(func=cmd_train)

    logger.debug("Registering subcommand 'doctor' from 'neuralops.cli.doctor'")
    from neuralops.cli.doctor import cmd_doctor

    doctor_p = subparsers.add_parser("doctor", help="Inspect local hardware, toolchains, and environment health")
    doctor_p.add_argument("--json", action="store_true", default=False, help="Output diagnostics as JSON")
    doctor_p.set_defaults(func=cmd_doctor)

    logger.debug("Registering subcommand 'info' from 'neuralops.cli.info'")
    from neuralops.cli.info import cmd_info

    info_p = subparsers.add_parser("info", help="Display hardware, framework, and registered component status")
    info_p.add_argument("--json", action="store_true", default=False, help="Output system information as JSON")
    info_p.set_defaults(func=cmd_info)

    logger.debug("Registering subcommands 'surgery' and 'transplant' from 'neuralops.cli.surgery_cmd'")
    from neuralops.cli.surgery_cmd import build_surgery_parser, build_transplant_parser, cmd_surgery, cmd_transplant

    surg_p = build_surgery_parser(
        subparsers.add_parser(
            "surgery",
            aliases=["recipe"],
            help="Plan, validate, and execute declarative transformation recipes",
        )
    )
    surg_p.set_defaults(func=cmd_surgery)

    trans_p = build_transplant_parser(
        subparsers.add_parser(
            "transplant",
            help="Surgically transplant attention mechanisms into model checkpoints",
        )
    )
    trans_p.set_defaults(func=cmd_transplant)

    logger.debug("Registering subcommand 'ui' from 'neuralops.ui.cli'")
    from neuralops.ui.cli import build_ui_parser, cmd_ui

    ui_p = build_ui_parser(
        subparsers.add_parser(
            "ui",
            help="Launch NeuralOps Web UI",
        )
    )
    ui_p.set_defaults(func=cmd_ui)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main execution router for NeuralOps CLI.

    Dispatches execution strictly using the standard `func` handler contract.
    """
    parser = build_main_parser()
    raw_args = sys.argv[1:] if argv is None else list(argv)

    if not raw_args or raw_args in (["-h"], ["--help"]):
        parser.print_help()
        return 0

    args = parser.parse_args(raw_args)

    # Dispatch strictly to attached `func` handler
    if hasattr(args, "func") and args.func is not None:
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
