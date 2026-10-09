"""Unified CLI for neuralops benchmarking suite."""
from __future__ import annotations

import argparse
import platform
import sys

from neuralops.benchmarks.api import Runner
from neuralops.benchmarks.discovery import discover_all_modules
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralops.benchmarks",
        description="Unified benchmarking tool for 232 attention & compressor modules.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    # 1. run
    run_p = subparsers.add_parser("run", help="Run benchmark(s)")
    run_p.add_argument("--module", type=str, default=None, help="Single module name")
    run_p.add_argument("--modules", type=str, default=None, help="Comma-separated module names")
    run_p.add_argument("--all-attention", action="store_true", help="Run all 159 attention modules")
    run_p.add_argument("--all-compressors", action="store_true", help="Run all 73 compressors")
    run_p.add_argument("--all", action="store_true", help="Run all 232 modules")
    run_p.add_argument("--no-compression", action="store_true", help="Run baseline uncompressed model")
    run_p.add_argument("--model", type=str, default="distilbert-base-uncased", help="Target model")
    run_p.add_argument("--models", type=str, default=None, help="Comma-separated target models")
    run_p.add_argument("--all-models", action="store_true", help="Benchmark all supported models")
    run_p.add_argument("--dataset", type=str, default="sst2", help="Target dataset")
    run_p.add_argument("--datasets", type=str, default=None, help="Comma-separated target datasets")
    run_p.add_argument("--all-datasets", action="store_true", help="Benchmark all supported datasets")
    run_p.add_argument("--rank", type=int, default=64, help="Rank for low-rank attention/compressor")
    run_p.add_argument("--ranks", type=str, default=None, help="Comma-separated ranks")
    run_p.add_argument("--device", type=str, default="cuda", help="Hardware device (cuda/mps/cpu)")
    run_p.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    run_p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    run_p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    run_p.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    run_p.add_argument("--parallel", type=int, default=1, help="Parallel worker count")
    run_p.add_argument("--output-dir", type=str, default="results", help="Directory to store outputs")
    run_p.add_argument("--timeout-s", type=int, default=1800, help="Timeout in seconds")
    run_p.add_argument("--eval-only", action="store_true", help="Skip fine-tuning")
    # Universal engine flags
    run_p.add_argument("--task", type=str, default=None, help="Target task type (e.g., classification, summarization, qa, lm)")
    run_p.add_argument("--max-samples", type=int, default=None, help="Limit number of dataset samples")
    run_p.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code for HF models")
    run_p.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for gated/private models")
    run_p.add_argument("--language", type=str, default=None, help="Language code for multilingual datasets/metrics")
    run_p.add_argument("--text-column", type=str, default=None, help="Custom column name for text inputs")
    run_p.add_argument("--label-column", type=str, default=None, help="Custom column name for target labels")
    run_p.add_argument("--id-column", type=str, default=None, help="Custom column name for sample IDs")
    run_p.add_argument("--preprocess-script", type=str, default=None, help="Path to custom preprocessing python script")
    run_p.add_argument("--config", type=str, default=None, help="Path to YAML benchmark configuration file")
    # Results management flags
    run_p.add_argument("--auto-commit", action="store_true", help="Auto-commit benchmark results to git")
    run_p.add_argument("--no-auto-commit", action="store_true", help="Disable git auto-commit")
    run_p.add_argument("--commit-interval", type=int, default=1, help="Interval of runs between git commits")
    run_p.add_argument("--push-to-hf", type=str, default=None, help="Hugging Face Hub repository ID to upload results")
    run_p.add_argument("--no-deduplicate", action="store_true", help="Disable run deduplication check")
    run_p.add_argument("--force-rerun", action="store_true", help="Force rerun even if identical run exists")

    # 2. resume
    res_p = subparsers.add_parser("resume", help="Resume interrupted sweep")
    res_p.add_argument("--output-dir", type=str, default="results", help="Output directory of interrupted run")
    res_p.add_argument("--run-id", type=str, default=None, help="Run ID to resume")

    # 3. plan
    plan_p = subparsers.add_parser("plan", help="Dry-run and estimate execution grid")
    plan_p.add_argument("--module", type=str, default=None)
    plan_p.add_argument("--modules", type=str, default=None)
    plan_p.add_argument("--all-attention", action="store_true")
    plan_p.add_argument("--all-compressors", action="store_true")
    plan_p.add_argument("--all", action="store_true")
    plan_p.add_argument("--model", type=str, default="distilbert-base-uncased")
    plan_p.add_argument("--models", type=str, default=None)
    plan_p.add_argument("--all-models", action="store_true")
    plan_p.add_argument("--dataset", type=str, default="sst2")
    plan_p.add_argument("--datasets", type=str, default=None)
    plan_p.add_argument("--all-datasets", action="store_true")
    plan_p.add_argument("--rank", type=int, default=64)
    plan_p.add_argument("--ranks", type=str, default=None)
    plan_p.add_argument("--yes", action="store_true", help="Confirm execution without prompt")

    # 4. list
    list_p = subparsers.add_parser("list", help="List registered modules with capability filters")
    list_p.add_argument("--type", choices=["attention", "compressor", "all"], default="all")
    list_p.add_argument("--supports", choices=["causal", "gqa", "rope", "cuda"], default=None)

    # 5. verify
    subparsers.add_parser("verify", help="Check system hardware and package requirements")

    # 6. results
    res_show = subparsers.add_parser("results", help="Inspect and summarize benchmark results")
    res_show.add_argument("--output-dir", type=str, default="results", help="Output directory")
    res_show.add_argument("--summary", action="store_true", help="Display summary table")
    res_show.add_argument("--latest", action="store_true", help="Show latest result details")

    return parser


def handle_list(args: argparse.Namespace) -> None:
    modules = discover_all_modules()
    if args.type in ("attention", "compressor"):
        modules = [m for m in modules if m.type == args.type]

    if args.supports:
        flag = f"supports_{args.supports}" if args.supports != "cuda" else "requires_cuda"
        modules = [m for m in modules if getattr(m, flag, False)]

    print(f"Discovered {len(modules)} matching modules:")
    for m in modules:
        print(f"  [{m.type:10s}] {m.name:36s} (causal={m.supports_causal}, gqa={m.supports_gqa}, rope={m.supports_rope}, cuda={m.requires_cuda})")


def handle_plan(args: argparse.Namespace) -> None:
    # Resolve modules count
    mod_count = 1
    if args.all:
        mod_count = len(ATTENTION_REGISTRY) + len(COMPRESSOR_REGISTRY)
    elif args.all_attention:
        mod_count = len(ATTENTION_REGISTRY)
    elif args.all_compressors:
        mod_count = len(COMPRESSOR_REGISTRY)
    elif args.modules:
        mod_count = len(args.modules.split(","))

    models_count = 10 if args.all_models else (len(args.models.split(",")) if args.models else 1)
    datasets_count = 22 if args.all_datasets else (len(args.datasets.split(",")) if args.datasets else 1)
    ranks_count = len(args.ranks.split(",")) if args.ranks else 1

    total_runs = mod_count * models_count * datasets_count * ranks_count
    est_seconds = total_runs * 180  # Assume ~3 mins per run average on 1 GPU
    est_hours = est_seconds / 3600.0

    print("═════════════════════════════════════════════════════")
    print(f"Benchmark Plan Summary:")
    print(f"  Modules:  {mod_count}")
    print(f"  Models:   {models_count}")
    print(f"  Datasets: {datasets_count}")
    print(f"  Ranks:    {ranks_count}")
    print(f"  Total planned runs: {total_runs}")
    print(f"  Estimated duration: ~{est_hours:.1f} hours on 1 GPU")
    print("═════════════════════════════════════════════════════")

    if total_runs > 500 and not args.yes:
        print("⚠️  Grid size exceeds 500 runs. Use --yes to confirm.")


def handle_verify(args: argparse.Namespace) -> None:
    import torch
    import neuralops.attentions
    import neuralops.compression_engine
    print("System Environment Verification:")
    print(f"  Platform:         {platform.platform()}")
    print(f"  Python Version:   {platform.python_version()}")
    print(f"  PyTorch Version:  {torch.__version__}")
    print(f"  CUDA Available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Name:      {torch.cuda.get_device_name(0)}")
        print(f"  Device Count:     {torch.cuda.device_count()}")
    print(f"  MPS Available:    {torch.backends.mps.is_available()}")
    print(f"  Attention Reg:    {len(ATTENTION_REGISTRY)} modules")
    print(f"  Compressor Reg:   {len(COMPRESSOR_REGISTRY)} modules")



def handle_results(args: argparse.Namespace) -> None:
    runner = Runner(output_dir=args.output_dir)
    df = runner.summary()
    if df.empty:
        print(f"No results found in {args.output_dir}")
        return
    print(df.to_string(index=False))


def handle_run(args: argparse.Namespace) -> None:
    mods = "all"
    if args.no_compression:
        mods = "baseline"
    elif args.module:
        mods = [args.module]
    elif args.modules:
        mods = [m.strip() for m in args.modules.split(",")]
    elif args.all_attention:
        mods = "all-attention"
    elif args.all_compressors:
        mods = "all-compressors"

    ranks = [int(r.strip()) for r in args.ranks.split(",")] if args.ranks else args.rank
    models = [m.strip() for m in args.models.split(",")] if args.models else args.model
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else args.dataset

    runner = Runner(
        modules=mods,
        models=models,
        datasets=datasets,
        ranks=ranks,
        parallel=args.parallel,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        device=args.device,
        timeout_s=args.timeout_s,
        eval_only=args.eval_only,
        task=args.task,
        max_samples=args.max_samples,
        trust_remote_code=not args.no_trust_remote_code,
        hf_token=args.hf_token,
        language=args.language,
        text_column=args.text_column,
        label_column=args.label_column,
        id_column=args.id_column,
        preprocess_script=args.preprocess_script,
        auto_commit=args.auto_commit and not args.no_auto_commit,
        commit_interval=args.commit_interval,
        push_to_hf=args.push_to_hf,
        deduplicate=not args.no_deduplicate,
        force_rerun=args.force_rerun,
    )
    for res in runner.stream():
        status_symbol = "[+]" if res.status == "success" else "[-]"
        mod_name = res.module.get("name") if isinstance(res.module, dict) else res.module
        mdl_name = res.model.get("name") if isinstance(res.model, dict) else res.model
        ds_name = res.dataset.get("name") if isinstance(res.dataset, dict) else res.dataset
        print(f"{status_symbol} [{res.status.upper()}] {mod_name} on {mdl_name}/{ds_name}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "list":
        handle_list(args)
    elif args.subcommand == "plan":
        handle_plan(args)
    elif args.subcommand == "verify":
        handle_verify(args)
    elif args.subcommand == "results":
        handle_results(args)
    elif args.subcommand == "run":
        handle_run(args)
    elif args.subcommand == "resume":
        runner = Runner(output_dir=args.output_dir, resume=True)
        runner.resume()

    return 0


if __name__ == "__main__":
    sys.exit(main())
