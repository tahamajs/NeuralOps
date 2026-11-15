"""Main CLI entrypoint for NeuralOps CLI v2.

Unified command-line interface for NeuralOps benchmark execution, planning,
diagnostics, results exploration, and web UI.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from typing import List, Optional

from neuralops._version import __version__
from neuralops.benchmarks.cli_v2.commands.benchmark import handle_benchmark, setup_benchmark_parser
from neuralops.benchmarks.cli_v2.commands.doctor import handle_doctor, setup_doctor_parser
from neuralops.benchmarks.cli_v2.commands.plan import handle_plan, setup_plan_parser
from neuralops.benchmarks.cli_v2.commands.results import handle_results, setup_results_parser
from neuralops.benchmarks.cli_v2.commands.ui import handle_ui, setup_ui_parser
from neuralops.benchmarks.cli_v2.utils.completion import generate_completion_script
from neuralops.benchmarks.cli_v2.utils.rendering import print_cli_error

LEGACY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "matrix",
        "multiseed",
        "compare",
        "list",
        "info",
        "resume",
        "verify",
        "export",
        "search",
        "compress",
        "generate",
        "niah",
        "benchmark-fused",
        "transplant",
        "surgery",
        "compose",
        "roofline",
        "recommend",
        "calibrate",
        "inspect",
        "tui",
        "validate",
        "scaffold",
        "ledger",
        "train",
        "eval",
        "evaluate",
        "serve",
        "monitor",
        "distill",
        "test",
        "recipe",
        "evidence",
        "ops",
        "compat",
    }
)


def _setup_completion_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for completion subcommand."""
    parser = subparsers.add_parser("completion", help="Generate shell autocompletion script")
    parser.add_argument("shell", choices=["bash", "zsh", "fish"], help="Target shell")
    return parser


def create_parser() -> argparse.ArgumentParser:
    """Create top-level parser for CLI v2.

    Returns:
        Configured ArgumentParser with all v2 subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="NeuralOps Benchmark CLI v2",
    )
    parser.add_argument("--version", action="version", version=f"NeuralOps {__version__}")
    parser.add_argument("--device", type=str, default=None, help="Global execution device (cpu, cuda, etc.)")
    parser.add_argument(
        "--install-completion",
        type=str,
        choices=["bash", "zsh", "fish"],
        default=None,
        help="Generate shell autocomplete script for bash, zsh, or fish",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    setup_benchmark_parser(subparsers)
    # Alias 'run' to benchmark parser for backward compatibility
    run_parser = subparsers.add_parser("run", help="Run benchmark suite (alias for benchmark)")
    run_parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON benchmark configuration")
    run_parser.add_argument("--modules", type=str, default="all", help="Modules to benchmark")
    run_parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda, etc.)")
    run_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    run_parser.add_argument("--output", type=str, default="results", help="Output directory")
    run_parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    run_parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    run_parser.add_argument("--dry-run", action="store_true", help="Dry run estimation without heavy execution")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")

    setup_plan_parser(subparsers)
    setup_results_parser(subparsers)
    setup_doctor_parser(subparsers)
    setup_ui_parser(subparsers)
    _setup_completion_parser(subparsers)
    return parser


def _handle_legacy_dispatch(argv: List[str]) -> int:
    """Delegate to legacy CLI parser and dispatcher for backward compatibility.

    Args:
        argv: Command-line arguments to pass to legacy dispatcher.

    Returns:
        Integer exit code.
    """
    import warnings

    subcmd = argv[0] if argv else "unknown"
    warnings.warn(
        f"The '{subcmd}' subcommand is deprecated in NeuralOps CLI and will be migrated to CLI v2. "
        "Please use canonical CLI v2 commands ('neuralops benchmark', 'neuralops plan', 'neuralops doctor', etc.).",
        DeprecationWarning,
        stacklevel=3,
    )
    from neuralops.benchmarks.cli import _build_parser, _dispatch

    legacy_parser = _build_parser()
    legacy_args = legacy_parser.parse_args(argv)
    ret = _dispatch(legacy_args)
    return 0 if ret is None else int(ret)


def _apply_device_env(raw_args: List[str]) -> None:
    """Detect --device cpu in raw args and configure environment accordingly."""
    if "--device" in raw_args:
        try:
            idx = raw_args.index("--device")
            if idx + 1 < len(raw_args) and raw_args[idx + 1] == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
        except ValueError:  # allowed-swallow: env var parsing fallback
            pass
    for arg in raw_args:
        if arg.startswith("--device=") and arg.split("=", 1)[1] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI v2 entrypoint.

    Args:
        argv: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for error, 130 for SIGINT).
    """
    raw_args = sys.argv[1:] if argv is None else list(argv)

    if not raw_args or raw_args in (["-h"], ["--help"]):
        try:
            from neuralops.cli.terminal_ui import print_hf_style_help

            print_hf_style_help()
            return 0
        except Exception:  # allowed-swallow: best-effort terminal UI display
            pass

    _apply_device_env(raw_args)

    if "--install-completion" in raw_args:
        try:
            idx = raw_args.index("--install-completion")
            if idx + 1 < len(raw_args):
                shell = raw_args[idx + 1]
                print(generate_completion_script(shell))
                return 0
        except (ValueError, IndexError):  # allowed-swallow: CLI argument parsing fallback
            pass

    # Check if non-flag tokens represent a legacy subcommand
    non_flag_tokens = [t for t in raw_args if not t.startswith("-")]
    if non_flag_tokens:
        if non_flag_tokens[0] in LEGACY_SUBCOMMANDS:
            return _handle_legacy_dispatch(raw_args)
        if (
            non_flag_tokens[0] in ("benchmark", "run")
            and len(non_flag_tokens) > 1
            and non_flag_tokens[1] in LEGACY_SUBCOMMANDS
        ):
            return _handle_legacy_dispatch(raw_args)

    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)

    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if getattr(args, "install_completion", None):
        print(generate_completion_script(args.install_completion))
        return 0

    try:
        if args.command in ("benchmark", "run"):
            return handle_benchmark(args)
        elif args.command == "plan":
            return handle_plan(args)
        elif args.command == "results":
            return handle_results(args)
        elif args.command == "doctor":
            return handle_doctor(args)
        elif args.command == "ui":
            return handle_ui(args)
        elif args.command == "completion":
            print(generate_completion_script(args.shell))
            return 0
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print_cli_error(f"File not found: {exc}", hint="Check file paths.")
        return 1
    except KeyError as exc:
        print_cli_error(f"Unrecognized key or identifier: {exc}")
        return 1
    except Exception as exc:
        print_cli_error(f"Command execution failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
