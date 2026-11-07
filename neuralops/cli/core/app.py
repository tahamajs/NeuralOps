#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Modular CLI Application Parser and Dispatcher for NeuralOps."""

from __future__ import annotations

import argparse
import sys

from neuralops import __version__
from neuralops.cli.core.commands.benchmark import handle_plan, handle_resume, handle_run
from neuralops.cli.core.commands.model import (
    handle_calibrate,
    handle_compress,
    handle_distill,
    handle_eval,
    handle_export,
    handle_generate,
    handle_search,
    handle_serve,
    handle_train,
    handle_transplant,
)
from neuralops.cli.core.commands.module import handle_inspect, handle_list, handle_scaffold, handle_validate
from neuralops.cli.core.commands.results import (
    handle_evidence,
    handle_ledger,
    handle_ops,
    handle_recipe,
    handle_results,
)
from neuralops.cli.core.commands.system import handle_compat, handle_doctor, handle_info, handle_ui, handle_verify
from neuralops.cli.core.common import add_benchmark_options, add_common_options
from neuralops.cli.core.errors import CliError, ExitCode, emit_cli_error


def build_app_parser() -> argparse.ArgumentParser:
    """Build the unified command parser with both grouped families and backward-compatible top-level aliases."""
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="Unified CLI & benchmarking suite for 203 attention and compressor modules.",
    )
    parser.add_argument("--version", "-v", action="version", version=f"NeuralOps v{__version__}")
    add_common_options(parser)

    subparsers = parser.add_subparsers(dest="subcommand", help="Command family or operation")

    # -------------------------------------------------------------
    # 1. Benchmark Family
    # -------------------------------------------------------------
    bench_p = subparsers.add_parser("benchmark", aliases=["run"], help="Benchmark execution, planning, and resumption")
    add_common_options(bench_p)
    add_benchmark_options(bench_p)

    plan_p = subparsers.add_parser("plan", help="Dry-run preview of planned benchmark execution grid")
    add_common_options(plan_p)
    add_benchmark_options(plan_p)

    resume_p = subparsers.add_parser("resume", help="Resume interrupted benchmark run from output directory")
    add_common_options(resume_p)
    resume_p.add_argument("--output-dir", type=str, default="results/latest", help="Output directory to resume from")

    # -------------------------------------------------------------
    # 2. Module Family
    # -------------------------------------------------------------
    module_p = subparsers.add_parser("module", help="Module exploration, inspection, validation, and scaffolding")
    mod_sub = module_p.add_subparsers(dest="module_action", help="Module action")

    list_sub = mod_sub.add_parser("list", help="List registered attention mechanisms and compressors")
    add_common_options(list_sub)
    list_sub.add_argument("--category", choices=["attention", "compressor", "cuda", "triton", "all"], default="all")
    list_sub.add_argument("--family", type=str, default=None)

    inspect_sub = mod_sub.add_parser("inspect", help="Inspect module specifications and quickstart code")
    add_common_options(inspect_sub)
    inspect_sub.add_argument("name", type=str, help="Module canonical key")
    inspect_sub.add_argument("--code-only", action="store_true", default=False)
    inspect_sub.add_argument("--roofline", action="store_true", default=False)

    validate_sub = mod_sub.add_parser("validate", help="Validate module numerical stability and causal gates")
    add_common_options(validate_sub)
    validate_sub.add_argument("name", type=str, help="Module canonical key to validate")
    validate_sub.add_argument(
        "--gate", choices=["forward", "backward", "causal", "parity", "dispatch", "all"], default="all"
    )

    scaffold_sub = mod_sub.add_parser("scaffold", help="Scaffold boilerplate code for a new module")
    add_common_options(scaffold_sub)
    scaffold_sub.add_argument("name", type=str, help="Module name")
    scaffold_sub.add_argument("--category", choices=["attention", "compressor"], default="attention")
    scaffold_sub.add_argument("--family", type=str, default="general")
    scaffold_sub.add_argument("--output", type=str, default=None)

    # Top-level module aliases for backwards compatibility
    list_top = subparsers.add_parser("list", help="Alias for 'module list'")
    add_common_options(list_top)
    list_top.add_argument("--category", choices=["attention", "compressor", "cuda", "triton", "all"], default="all")
    list_top.add_argument("--family", type=str, default=None)

    inspect_top = subparsers.add_parser("inspect", help="Alias for 'module inspect'")
    add_common_options(inspect_top)
    inspect_top.add_argument("name", type=str, help="Module canonical key")
    inspect_top.add_argument("--code-only", action="store_true", default=False)
    inspect_top.add_argument("--roofline", action="store_true", default=False)

    validate_top = subparsers.add_parser("validate", help="Alias for 'module validate'")
    add_common_options(validate_top)
    validate_top.add_argument("name", type=str, help="Module canonical key")
    validate_top.add_argument(
        "--gate", choices=["forward", "backward", "causal", "parity", "dispatch", "all"], default="all"
    )

    scaffold_top = subparsers.add_parser("scaffold", help="Alias for 'module scaffold'")
    add_common_options(scaffold_top)
    scaffold_top.add_argument("name", type=str, help="Module name")
    scaffold_top.add_argument("--category", choices=["attention", "compressor"], default="attention")
    scaffold_top.add_argument("--family", type=str, default="general")
    scaffold_top.add_argument("--output", type=str, default=None)

    # -------------------------------------------------------------
    # 3. Model Family
    # -------------------------------------------------------------
    comp_p = subparsers.add_parser("compress", help="Compress model with low-rank SVD, pruning, or quantization")
    add_common_options(comp_p)
    comp_p.add_argument("--model", type=str, required=True, help="HuggingFace model ID or local path")
    comp_p.add_argument("--method", type=str, default="hnc_svd", help="Compression engine key")
    comp_p.add_argument("--rank", type=int, default=16, help="Target rank")
    comp_p.add_argument("--output", type=str, default="compressed_model", help="Save directory")

    trans_p = subparsers.add_parser(
        "transplant", aliases=["surgery"], help="Surgically transplant attention mechanisms"
    )
    add_common_options(trans_p)
    trans_p.add_argument("--model", type=str, required=True, help="Target model ID or path")
    trans_p.add_argument("--target", type=str, required=True, help="Target attention mechanism key")
    trans_p.add_argument("--rank", type=int, default=16, help="Target rank")
    trans_p.add_argument("--calibrate", action="store_true", default=False)
    trans_p.add_argument("--heal", action="store_true", default=False)
    trans_p.add_argument("--heal-steps", type=int, default=100)
    trans_p.add_argument("--heal-lr", type=float, default=5e-5)
    trans_p.add_argument("--reference-model", type=str, default=None)
    trans_p.add_argument("--output", type=str, default=None)

    calib_p = subparsers.add_parser("calibrate", help="Calibrate activation covariance or heal model after surgery")
    add_common_options(calib_p)
    calib_p.add_argument("--model", type=str, required=True)
    calib_p.add_argument("--dataset", type=str, default=None)
    calib_p.add_argument("--output", type=str, default=None)

    search_p = subparsers.add_parser("search", help="AutoSVD Pareto rank allocation search")
    add_common_options(search_p)
    search_p.add_argument("--model", type=str, required=True)
    search_p.add_argument("--target-param-ratio", type=float, default=0.5)
    search_p.add_argument("--min-rank", type=int, default=8)
    search_p.add_argument("--max-rank", type=int, default=128)
    search_p.add_argument("--output-json", type=str, default=None)
    search_p.add_argument("--save-compressed", type=str, default=None)

    exp_p = subparsers.add_parser("export", help="Export model to ONNX, GGUF, SafeTensors, or TorchScript")
    add_common_options(exp_p)
    exp_p.add_argument("--model", type=str, required=True)
    exp_p.add_argument("--format", choices=["onnx", "gguf", "safetensors", "torchscript"], default="safetensors")
    exp_p.add_argument("--attn", type=str, default=None)
    exp_p.add_argument("--comp", type=str, default=None)
    exp_p.add_argument("--rank", type=int, default=16)

    gen_p = subparsers.add_parser("generate", help="Autoregressive text generation with speculative decoding")
    add_common_options(gen_p)
    gen_p.add_argument("--model", type=str, required=True)
    gen_p.add_argument("--prompt", type=str, default="NeuralOps acceleration test:")
    gen_p.add_argument("--max-new-tokens", type=int, default=32)
    gen_p.add_argument("--temperature", type=float, default=0.7)
    gen_p.add_argument("--speculative", action="store_true", default=False)
    gen_p.add_argument("--draft-k", type=int, default=3)

    serve_p = subparsers.add_parser("serve", help="Launch inference serving server")
    add_common_options(serve_p)
    serve_p.add_argument("--model", type=str, required=True)
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--host", type=str, default="0.0.0.0")

    train_p = subparsers.add_parser("train", help="Launch model training")
    add_common_options(train_p)
    train_p.add_argument("--model", type=str, required=True)

    eval_p = subparsers.add_parser("eval", aliases=["evaluate"], help="Evaluate model accuracy and latency")
    add_common_options(eval_p)
    eval_p.add_argument("--model", type=str, required=True)

    distill_p = subparsers.add_parser("distill", help="Distill knowledge from teacher to student model")
    add_common_options(distill_p)
    distill_p.add_argument("--teacher", type=str, required=True)
    distill_p.add_argument("--student", type=str, required=True)
    distill_p.add_argument("--divergence", choices=["kl", "reverse_kl", "jsd", "forward_kl"], default="kl")
    distill_p.add_argument("--temperature", type=float, default=2.0)
    distill_p.add_argument("--alpha", type=float, default=0.5)
    distill_p.add_argument("--epochs", type=int, default=3)
    distill_p.add_argument("--batch-size", type=int, default=16)
    distill_p.add_argument("--lr", type=float, default=5e-5)
    distill_p.add_argument("--dataset", type=str, default="synthetic")

    # -------------------------------------------------------------
    # 4. System Family
    # -------------------------------------------------------------
    verify_p = subparsers.add_parser("verify", help="Hardware, PyTorch backend, and canonical counts verification")
    add_common_options(verify_p)

    doc_p = subparsers.add_parser("doctor", help="Environment diagnostics and dependency health check")
    add_common_options(doc_p)

    info_p = subparsers.add_parser("info", help="Environment telemetry and CUDA capability details")
    add_common_options(info_p)

    compat_p = subparsers.add_parser("compat", help="Portability matrix check across acceleration backends")
    add_common_options(compat_p)

    ui_p = subparsers.add_parser("ui", help="Launch interactive Terminal UI")
    ui_p.add_argument("--inspect", type=str, default=None)
    ui_p.add_argument("--snapshot", action="store_true", default=False)

    # -------------------------------------------------------------
    # 5. Results & Pipelines Family
    # -------------------------------------------------------------
    res_p = subparsers.add_parser(
        "results", aliases=["ledger"], help="Query and export benchmark results or SQLite ledger"
    )
    add_common_options(res_p)
    res_p.add_argument("--format", choices=["table", "markdown", "latex", "csv", "json"], default="table")
    res_p.add_argument("--dossier", type=str, default=None)
    res_p.add_argument("--diff", type=str, default=None)
    res_p.add_argument("--rev-a", type=int, default=None)
    res_p.add_argument("--rev-b", type=int, default=None)
    res_p.add_argument("--sync", action="store_true", default=False)
    res_p.add_argument("--delete", action="store_true", default=False)
    res_p.add_argument("--add", action="store_true", default=False)
    res_p.add_argument("--id", type=str, default=None)
    res_p.add_argument("--module", type=str, default=None)
    res_p.add_argument("--category", type=str, default=None)
    res_p.add_argument("--family", type=str, default=None)
    res_p.add_argument("--version-tag", type=str, default=None)
    res_p.add_argument("--hard", action="store_true", default=False)
    res_p.add_argument("--metrics", type=str, default=None)
    res_p.add_argument("--setup", type=str, default=None)
    res_p.add_argument("--dataset", type=str, default=None)
    res_p.add_argument("--backbone", type=str, default=None)
    res_p.add_argument("--method", type=str, default=None)
    res_p.add_argument("--mode", type=str, default="archive_old")
    res_p.add_argument("--all-revisions", action="store_true", default=False)
    res_p.add_argument("--limit", type=int, default=50)
    res_p.add_argument("--plot", choices=["scatter", "pareto", "bar", "history"], default=None)
    res_p.add_argument("--x-metric", type=str, default=None)
    res_p.add_argument("--y-metric", type=str, default=None)
    res_p.add_argument("--out", type=str, default=None)

    rec_p = subparsers.add_parser("recipe", help="Declarative optimization recipe commands")
    rec_sub = rec_p.add_subparsers(dest="recipe_action", help="Recipe action")
    for act in ["plan", "execute", "validate", "diff", "evidence", "benchmark"]:
        sp = rec_sub.add_parser(act)
        add_common_options(sp)
        sp.add_argument("file", type=str, nargs="?", default=None)
        if act == "diff":
            sp.add_argument("file1", type=str)
            sp.add_argument("file2", type=str)

    ev_p = subparsers.add_parser("evidence", help="Inspection of forensic execution evidence")
    ev_p.add_argument("evidence_action", choices=["show"], default="show")
    ev_p.add_argument("file", type=str)

    ops_p = subparsers.add_parser("ops", help="List registered operation contracts")
    ops_p.add_argument("ops_action", choices=["list"], default="list")

    return parser


def dispatch_command(args: argparse.Namespace) -> int:
    """Route parsed namespace to the matching handler."""
    sub = args.subcommand
    if sub in ("run", "benchmark"):
        return handle_run(args)
    elif sub == "plan":
        return handle_plan(args)
    elif sub == "resume":
        return handle_resume(args)
    elif sub == "module":
        action = getattr(args, "module_action", None)
        if action == "list":
            return handle_list(args)
        elif action == "inspect":
            return handle_inspect(args)
        elif action == "validate":
            return handle_validate(args)
        elif action == "scaffold":
            return handle_scaffold(args)
        return handle_list(args)
    elif sub == "list":
        return handle_list(args)
    elif sub == "inspect":
        return handle_inspect(args)
    elif sub == "validate":
        return handle_validate(args)
    elif sub == "scaffold":
        return handle_scaffold(args)
    elif sub in ("surgery", "transplant"):
        return handle_transplant(args)
    elif sub == "compress":
        return handle_compress(args)
    elif sub == "calibrate":
        return handle_calibrate(args)
    elif sub == "search":
        return handle_search(args)
    elif sub == "export":
        return handle_export(args)
    elif sub == "generate":
        return handle_generate(args)
    elif sub == "serve":
        return handle_serve(args)
    elif sub == "train":
        return handle_train(args)
    elif sub in ("eval", "evaluate"):
        return handle_eval(args)
    elif sub == "distill":
        return handle_distill(args)
    elif sub == "verify":
        return handle_verify(args)
    elif sub == "doctor":
        return handle_doctor(args)
    elif sub == "info":
        return handle_info(args)
    elif sub == "compat":
        return handle_compat(args)
    elif sub == "ui":
        return handle_ui(args)
    elif sub in ("results", "ledger"):
        return handle_results(args)
    elif sub == "recipe":
        return handle_recipe(args)
    elif sub == "evidence":
        return handle_evidence(args)
    elif sub == "ops":
        return handle_ops(args)

    return int(ExitCode.USAGE_ERROR)


def app_main(argv: list[str] | None = None) -> int:
    """Main entrypoint for NeuralOps CLI with structured exception and exit code discipline."""
    parser = build_app_parser()

    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args or raw_args in (["-h"], ["--help"]):
        parser.print_help()
        return int(ExitCode.SUCCESS)

    try:
        args = parser.parse_args(argv)
        if not args.subcommand:
            parser.print_help()
            return int(ExitCode.SUCCESS)
        return dispatch_command(args)
    except KeyboardInterrupt:
        sys.stderr.write("\n[Execution interrupted by user]\n")
        return int(ExitCode.USER_INTERRUPT)
    except CliError as err:
        emit_cli_error(err)
        return int(err.exit_code)
    except Exception as err:
        err_type = type(err).__name__
        msg = f"Unexpected failure ({err_type}): {err}"
        emit_cli_error(msg, exit_code=ExitCode.INTERNAL_ERROR)
        return int(ExitCode.INTERNAL_ERROR)
