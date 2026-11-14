"""Main CLI entrypoint for NeuralOps CLI v2."""

import argparse
import os
import sys
from typing import List, Optional

from neuralops.benchmarks.cli_v2.commands.benchmark import handle_benchmark, setup_benchmark_parser
from neuralops.benchmarks.cli_v2.commands.doctor import handle_doctor, setup_doctor_parser
from neuralops.benchmarks.cli_v2.commands.plan import handle_plan, setup_plan_parser
from neuralops.benchmarks.cli_v2.commands.results import handle_results, setup_results_parser
from neuralops.benchmarks.cli_v2.commands.ui import handle_ui, setup_ui_parser


def create_parser() -> argparse.ArgumentParser:
    """Create top-level parser for CLI v2."""
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="NeuralOps Benchmark CLI v2",
    )
    parser.add_argument("--device", type=str, default=None, help="Global execution device (cpu, cuda, etc.)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup_benchmark_parser(subparsers)
    setup_plan_parser(subparsers)
    setup_results_parser(subparsers)
    setup_doctor_parser(subparsers)
    setup_ui_parser(subparsers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI v2 entrypoint."""
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if "--device" in raw_args:
        try:
            idx = raw_args.index("--device")
            if idx + 1 < len(raw_args) and raw_args[idx + 1] == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
        except ValueError:
            pass
    for arg in raw_args:
        if arg.startswith("--device=") and arg.split("=", 1)[1] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = create_parser()
    args = parser.parse_args(argv)

    # Global or subcommand --device cpu handling
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.command == "benchmark":
        return handle_benchmark(args)
    elif args.command == "plan":
        return handle_plan(args)
    elif args.command == "results":
        return handle_results(args)
    elif args.command == "doctor":
        return handle_doctor(args)
    elif args.command == "ui":
        return handle_ui(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
