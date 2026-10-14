"""Unified CLI for neuralops benchmarking suite."""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import platform
import sys

import torch

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
    # Module selection shortcuts
    grp_run = run_p.add_argument_group("Module selection shortcuts")
    grp_run.add_argument("--all-attn", "--all-attention", dest="all_attn", action="store_true", help="Run every registered attention module")
    grp_run.add_argument("--all-comp", "--all-compressors", dest="all_comp", action="store_true", help="Run every registered compressor module")
    grp_run.add_argument("--all-delta", "--delta", dest="all_delta", action="store_true", help="Run all delta rule modules")
    grp_run.add_argument("--all-gated", "--gated", dest="all_gated", action="store_true", help="Run all gated attention modules")
    grp_run.add_argument("--all-svd", "--svd", dest="all_svd", action="store_true", help="Run all SVD-based modules (attention + compressor)")
    grp_run.add_argument("--all-ssm", "--ssm", dest="all_ssm", action="store_true", help="Run all SSM / state-space modules")
    grp_run.add_argument("--all-linear", "--linear", dest="all_linear", action="store_true", help="Run all linear attention modules")
    grp_run.add_argument("--all-sparse", "--sparse", dest="all_sparse", action="store_true", help="Run all sparse attention modules")
    grp_run.add_argument("--all-hybrid", "--hybrid", dest="all_hybrid", action="store_true", help="Run all hybrid attention modules")
    grp_run.add_argument("--all-modules", "--all", dest="all_modules", action="store_true", help="Run EVERY registered module")
    grp_run.add_argument("--fail-delta", dest="fail_delta", action="store_true", help="Run the 25 FAIL_DELTA modules from the audit report")
    grp_run.add_argument("--errors-only", dest="errors_only", action="store_true", help="Run only modules that previously errored")
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
    # Universal engine & training type flags
    run_p.add_argument("--training-type", type=str, default="finetune", choices=["finetune", "kd", "recovery", "lora", "none"], help="Training regime: finetune, kd, recovery, lora, or none")
    run_p.add_argument("--kd-teacher", type=str, default=None, help="Teacher model name for knowledge distillation")
    run_p.add_argument("--kd-temperature", type=float, default=2.0, help="Softmax temperature for KD")
    run_p.add_argument("--kd-alpha", type=float, default=0.5, help="Balance between task loss and distillation loss")
    run_p.add_argument("--recovery-steps", type=int, default=50, help="Number of recovery steps after compression")
    run_p.add_argument("--max-samples", type=int, default=None, help="Limit training set to N samples. Default: full dataset. Use 5000 for fast sweeps on slow GPUs.")
    run_p.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval set to N samples.")
    run_p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    # MLOps & Distributed Training
    run_p.add_argument("--tracker", choices=["none", "wandb", "mlflow", "tensorboard", "all"], default="none", help="Experiment tracking backend")
    run_p.add_argument("--project-name", type=str, default="neuralops", help="MLOps project/experiment name")
    run_p.add_argument("--run-name", type=str, default=None, help="MLOps run name")
    run_p.add_argument("--log-interval", type=int, default=10, help="Logging step interval")
    run_p.add_argument("--distributed-backend", choices=["none", "ddp", "fsdp", "deepspeed"], default="none", help="Distributed training backend")
    run_p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no", help="Mixed precision mode")
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
    # Module selection shortcuts
    grp_plan = plan_p.add_argument_group("Module selection shortcuts")
    grp_plan.add_argument("--all-attn", "--all-attention", dest="all_attn", action="store_true", help="Run every registered attention module")
    grp_plan.add_argument("--all-comp", "--all-compressors", dest="all_comp", action="store_true", help="Run every registered compressor module")
    grp_plan.add_argument("--all-delta", "--delta", dest="all_delta", action="store_true", help="Run all delta rule modules")
    grp_plan.add_argument("--all-gated", "--gated", dest="all_gated", action="store_true", help="Run all gated attention modules")
    grp_plan.add_argument("--all-svd", "--svd", dest="all_svd", action="store_true", help="Run all SVD-based modules (attention + compressor)")
    grp_plan.add_argument("--all-ssm", "--ssm", dest="all_ssm", action="store_true", help="Run all SSM / state-space modules")
    grp_plan.add_argument("--all-linear", "--linear", dest="all_linear", action="store_true", help="Run all linear attention modules")
    grp_plan.add_argument("--all-sparse", "--sparse", dest="all_sparse", action="store_true", help="Run all sparse attention modules")
    grp_plan.add_argument("--all-hybrid", "--hybrid", dest="all_hybrid", action="store_true", help="Run all hybrid attention modules")
    grp_plan.add_argument("--all-modules", "--all", dest="all_modules", action="store_true", help="Run EVERY registered module")
    grp_plan.add_argument("--fail-delta", dest="fail_delta", action="store_true", help="Run the 25 FAIL_DELTA modules from the audit report")
    grp_plan.add_argument("--errors-only", dest="errors_only", action="store_true", help="Run only modules that previously errored")
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

    # 7. export
    exp_p = subparsers.add_parser("export", help="Export model to production artifact format")
    exp_p.add_argument("--model", type=str, required=True, help="Base model identifier or path")
    exp_p.add_argument("--attn", "--attention", dest="attn", type=str, default=None, help="Attention mechanism to inject")
    exp_p.add_argument("--comp", "--compressor", dest="comp", type=str, default=None, help="Compression engine to apply")
    exp_p.add_argument("--rank", type=int, default=64, help="Rank for attention or compression")
    exp_p.add_argument("--format", choices=["safetensors", "onnx", "torchscript"], default="safetensors", help="Export format")
    exp_p.add_argument("--output-dir", type=str, default="exported_model", help="Directory for exported artifacts")

    # 8. doctor
    subparsers.add_parser("doctor", help="Run comprehensive diagnostics on environment and registrations")

    # 9. info
    subparsers.add_parser("info", help="Print system, GPU, and registry summary information")

    # 10. search
    search_p = subparsers.add_parser("search", help="Automated Fisher-information & Pareto rank allocation search (AutoSVD)")
    search_p.add_argument("--model", type=str, required=True, help="Base model identifier or path")
    search_p.add_argument("--target-param-ratio", type=float, default=0.5, help="Target compression parameter ratio (e.g. 0.5 for 50 percent)")
    search_p.add_argument("--min-rank", type=int, default=16, help="Minimum allowable rank per linear layer")
    search_p.add_argument("--max-rank", type=int, default=None, help="Maximum allowable rank per linear layer")
    search_p.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for allocated ranks")
    search_p.add_argument("--save-compressed", type=str, default=None, help="Directory to save the compressed model")

    # 11. compress
    comp_p = subparsers.add_parser("compress", help="Compress a model using SVD, W4A16, Hadamard, or any registered compressor")
    comp_p.add_argument("--model", type=str, required=True, help="Base model name or path (e.g. gpt2, roberta-base)")
    comp_p.add_argument("--method", "--engine", dest="method", type=str, default="hadamard_incoherent_svd", help="Compression engine name")
    comp_p.add_argument("--rank", type=int, default=32, help="Target rank for low-rank SVD compression")
    comp_p.add_argument("--save-dir", "--output-dir", dest="save_dir", type=str, default=None, help="Directory to save compressed weights")

    # 12. generate
    gen_p = subparsers.add_parser("generate", help="Run accelerated generation with draft-and-verify self-speculative decoding")
    gen_p.add_argument("--model", type=str, required=True, help="Target model identifier or path")
    gen_p.add_argument("--prompt", type=str, default="NeuralOps is a cutting-edge deep learning systems library that", help="Input prompt text")
    gen_p.add_argument("--max-new-tokens", type=int, default=32, help="Number of new tokens to generate")
    gen_p.add_argument("--speculative", action="store_true", default=True, help="Enable self-speculative draft-and-verify decoding")
    gen_p.add_argument("--draft-k", type=int, default=4, help="Draft tokens per step")
    gen_p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")

    # 13. niah
    niah_p = subparsers.add_parser("niah", help="Run Needle-In-A-Haystack passkey retrieval benchmark and generate 2D heatmap")
    niah_p.add_argument("--model", type=str, required=True, help="Model identifier or path")
    niah_p.add_argument("--context-lengths", type=str, default="256,512,1024,2048", help="Comma-separated sequence lengths")
    niah_p.add_argument("--needle-depths", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated needle depths (0.0 to 1.0)")
    niah_p.add_argument("--output-plot", type=str, default="paper/figures/fig_niah_passkey_heatmap.png", help="Path to save 2D heatmap")

    # 14. benchmark-fused
    fuse_p = subparsers.add_parser("benchmark-fused", help="Benchmark fused Triton kernels (W4A16, RoPE, State-Update, Loss)")
    fuse_p.add_argument("--kernel", choices=["w4a16", "rope", "state_update", "loss", "all"], default="all", help="Target kernel to benchmark")
    fuse_p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    fuse_p.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    fuse_p.add_argument("--dim", type=int, default=768, help="Hidden dimension")

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
    from neuralops.benchmarks.orchestrator import BenchmarkOrchestrator
    orch = BenchmarkOrchestrator(
        modules=args.modules or args.module or "all",
        delta=getattr(args, "all_delta", False),
        gated=getattr(args, "all_gated", False),
        svd=getattr(args, "all_svd", False),
        ssm=getattr(args, "all_ssm", False),
        linear=getattr(args, "all_linear", False),
        sparse=getattr(args, "all_sparse", False),
        hybrid=getattr(args, "all_hybrid", False),
        all_attn=getattr(args, "all_attn", False),
        all_comp=getattr(args, "all_comp", False),
        all_modules=getattr(args, "all_modules", False),
        fail_delta=getattr(args, "fail_delta", False),
        errors_only=getattr(args, "errors_only", False),
    )
    mod_count = len(orch.modules)

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
    module_arg = getattr(args, "module", None)
    modules_arg = getattr(args, "modules", None)
    if args.no_compression:
        mods = "baseline"
    elif module_arg:
        mods = [module_arg]
    elif modules_arg:
        mods = [m.strip() for m in modules_arg.split(",")]

    ranks = [int(r.strip()) for r in args.ranks.split(",")] if args.ranks else args.rank
    models = [m.strip() for m in args.models.split(",")] if args.models else args.model
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else args.dataset

    runner = Runner(
        modules=mods,
        delta=getattr(args, "all_delta", False),
        gated=getattr(args, "all_gated", False),
        svd=getattr(args, "all_svd", False),
        ssm=getattr(args, "all_ssm", False),
        linear=getattr(args, "all_linear", False),
        sparse=getattr(args, "all_sparse", False),
        hybrid=getattr(args, "all_hybrid", False),
        all_attn=getattr(args, "all_attn", False),
        all_comp=getattr(args, "all_comp", False),
        all_modules=getattr(args, "all_modules", False),
        fail_delta=getattr(args, "fail_delta", False),
        errors_only=getattr(args, "errors_only", False),
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
        eval_only=(args.training_type == "none"),
        training_type=args.training_type,
        kd_teacher=args.kd_teacher,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        recovery_steps=args.recovery_steps,
        task=getattr(args, "task", None),
        max_samples=args.max_samples,
        max_eval_samples=args.max_eval_samples,
        grad_accum_steps=args.grad_accum_steps,
        tracker=args.tracker,
        project_name=args.project_name,
        run_name=args.run_name,
        log_interval=args.log_interval,
        distributed_backend=args.distributed_backend,
        mixed_precision=args.mixed_precision,
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


def handle_export(args: argparse.Namespace) -> None:
    from transformers import AutoModel
    import neuralops
    print(f"Loading base model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model)
    if args.attn:
        print(f"Applying attention replacement: '{args.attn}' (rank={args.rank})...")
        model = neuralops.replace_attention(model, attention=args.attn, rank=args.rank)
    if args.comp:
        print(f"Applying model compression: '{args.comp}' (rank={args.rank})...")
        model = neuralops.compress(model, method=args.comp, rank=args.rank)

    from neuralops.export.exporter import export_model
    out = export_model(
        model=model,
        output_dir=args.output_dir,
        format=args.format,
        model_name_or_path=args.model,
        attention_type=args.attn,
        compression_method=args.comp,
        rank=args.rank,
    )
    print(f"[+] Successfully exported model to {out}")


def handle_search(args: argparse.Namespace) -> int:
    """Executes AutoSVD Fisher & Gram sensitivity search and Pareto rank allocation."""
    import json
    from transformers import AutoModel
    from neuralops.compression_engine.svd.auto_svd import AutoSVDCompressor, solve_optimal_rank_allocation

    print(f"Loading model '{args.model}' for AutoSVD Pareto rank search...")
    model = AutoModel.from_pretrained(args.model)

    print(f"Solving optimal rank allocation for target_param_ratio={args.target_param_ratio} (min_rank={args.min_rank})...")
    ranks, stats = solve_optimal_rank_allocation(
        model,
        target_param_ratio=args.target_param_ratio,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )

    print("\n═════════════════════════════════════════════════════")
    print(f"AutoSVD Pareto Allocation Results for {args.model}:")
    print(f"  Original Params:   {stats.get('original_params', 0):,}")
    print(f"  Compressed Params: {stats.get('compressed_params', 0):,}")
    print(f"  Compression Ratio: {stats.get('compression_ratio', 1.0):.2f}x")
    print(f"  Param Savings:     {stats.get('param_savings_pct', 0.0):.1f}%")
    print(f"  Optimized Layers:  {stats.get('num_layers_optimized', 0)}")
    print("─────────────────────────────────────────────────────")
    for name, r in list(ranks.items())[:10]:
        print(f"  {name:40s} -> rank {r}")
    if len(ranks) > 10:
        print(f"  ... and {len(ranks) - 10} more layers.")
    print("═════════════════════════════════════════════════════\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"allocated_ranks": ranks, "stats": stats}, f, indent=2)
        print(f"[+] Saved allocated ranks JSON to {args.output_json}")

    if args.save_compressed:
        print(f"Applying compression and saving model to {args.save_compressed}...")
        compressor = AutoSVDCompressor(
            model,
            target_param_ratio=args.target_param_ratio,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
        )
        c_model, _ = compressor.compress()
        c_model.save_pretrained(args.save_compressed)
        print(f"[+] Compressed model successfully saved to {args.save_compressed}")

    return 0


def handle_compress(args: argparse.Namespace) -> int:
    """Executes model compression via the requested engine."""
    from transformers import AutoModel
    from neuralops.eval_all_models_datasets import apply_module_to_model

    print(f"Loading base model '{args.model}' for compression...")
    model = AutoModel.from_pretrained(args.model)
    orig_params = sum(p.numel() for p in model.parameters())

    print(f"Applying compression engine '{args.method}' (rank={args.rank})...")
    model, stats = apply_module_to_model(
        model,
        args.method,
        module_type="compressor",
        rank=args.rank,
        device=model.device,
        fallback_to_original=False,
    )
    new_params = sum(p.numel() for p in model.parameters())
    ratio = orig_params / max(1, new_params)
    savings = 100.0 * (1.0 - new_params / max(1, orig_params))

    print("\n═════════════════════════════════════════════════════")
    print(f"Compression Complete: {args.method} on {args.model}")
    print(f"  Original Parameters:   {orig_params:,}")
    print(f"  Compressed Parameters: {new_params:,}")
    print(f"  Compression Ratio:     {ratio:.2f}x")
    print(f"  Parameter Reduction:   {savings:.1f}%")
    print("═════════════════════════════════════════════════════\n")

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print(f"[+] Compressed model artifacts saved to '{args.save_dir}'")
    return 0


def handle_generate(args: argparse.Namespace) -> int:
    """Runs autoregressive text generation with self-speculative draft-and-verify decoding."""
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder

    print(f"Loading generative model '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Generating {args.max_new_tokens} tokens (speculative={args.speculative}, draft_k={args.draft_k})...\n")

    decoder = SelfSpeculativeDecoder(
        target_model=model,
        max_draft_k=args.draft_k,
        adaptive_k=True,
    )

    t0 = time.time()
    out_ids, stats = decoder.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0
    gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print("═════════════════════════════════════════════════════")
    print("Generated Text:")
    print(f"\"{gen_text}\"")
    print("─────────────────────────────────────────────────────")
    print(f"Tokens Generated:      {stats.total_tokens_generated}")
    print(f"Target Model Passes:   {stats.total_target_forward_passes}")
    print(f"Draft Acceptance Rate: {stats.acceptance_rate * 100:.1f}%")
    print(f"Mean Accepted / Step:  {stats.mean_accepted_tokens_per_step:.2f}")
    print(f"Throughput:            {stats.tokens_per_sec:.1f} tokens/s")
    print(f"Speculative Speedup:   {stats.speedup_over_standard:.2f}x")
    print("═════════════════════════════════════════════════════\n")
    return 0


def handle_niah(args: argparse.Namespace) -> int:
    """Runs Needle-In-A-Haystack retrieval evaluation and produces a 2D passkey heatmap."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuralops.evaluation.niah_passkey import NeedleInAHaystackEvaluator, plot_niah_heatmap

    print(f"Loading model '{args.model}' for Needle-In-A-Haystack Passkey evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    ctx_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    depths = [float(x.strip()) for x in args.needle_depths.split(",")]

    evaluator = NeedleInAHaystackEvaluator(
        model=model,
        tokenizer=tokenizer,
        context_lengths=ctx_lengths,
        needle_depths=depths,
        device=device,
    )

    print(f"Running NIAH benchmark across lengths={ctx_lengths} and depths={depths}...")
    matrix = evaluator.evaluate()

    print(f"Generating 2D heatmap -> '{args.output_plot}'...")
    fig = plot_niah_heatmap(
        matrix,
        context_lengths=ctx_lengths,
        depths=depths,
        save_path=args.output_plot,
        model_name=args.model,
    )
    print(f"[+] NIAH Passkey heatmap saved to '{args.output_plot}'\n")
    return 0


def handle_benchmark_fused(args: argparse.Namespace) -> int:
    """Benchmarks fused @triton.jit kernels against PyTorch eager baselines."""
    import time
    print(f"Benchmarking Fused Triton Kernels (kernel={args.kernel}, B={args.batch_size}, L={args.seq_len}, D={args.dim})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[-] Fused Triton benchmarks require a CUDA GPU. PyTorch fallbacks will be used.")

    # 1. W4A16 Benchmark
    if args.kernel in ("w4a16", "all"):
        from neuralops.core.triton_kernels.low_rank import BitPackedW4A16Linear
        w4 = BitPackedW4A16Linear(args.dim, args.dim, group_size=64, bias=False).to(device)
        x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=torch.float16)
        # Warmup
        for _ in range(5):
            _ = w4(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 50
        for _ in range(iters):
            _ = w4(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dur = (time.time() - t0) / iters * 1000
        print(f"  [W4A16 Dequant-GEMM]:  {dur:.2f} ms / forward pass")

    # 2. State-Update Benchmark
    if args.kernel in ("state_update", "all"):
        from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
        s = torch.zeros(args.batch_size, 4, 32, 32, device=device)
        q = torch.randn(args.batch_size, 4, 32, device=device)
        k = torch.randn(args.batch_size, 4, 32, device=device)
        v = torch.randn(args.batch_size, 4, 32, device=device)
        for _ in range(5):
            _ = fused_recurrent_linear_step(q, k, v, s)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 100
        for _ in range(iters):
            _ = fused_recurrent_linear_step(q, k, v, s)
        if device == "cuda":
            torch.cuda.synchronize()
        dur = (time.time() - t0) / iters * 1000
        print(f"  [Fused O(1) Linear Step]: {dur:.3f} ms / step")

    print("[+] Fused kernel benchmark completed successfully.\n")
    return 0


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
    elif args.subcommand == "export":
        handle_export(args)
    elif args.subcommand == "search":
        return handle_search(args)
    elif args.subcommand == "compress":
        return handle_compress(args)
    elif args.subcommand == "generate":
        return handle_generate(args)
    elif args.subcommand == "niah":
        return handle_niah(args)
    elif args.subcommand == "benchmark-fused":
        return handle_benchmark_fused(args)
    elif args.subcommand == "resume":
        runner = Runner(output_dir=args.output_dir, resume=True)
        runner.resume()
    elif args.subcommand == "doctor":
        from neuralops.cli.doctor import cmd_doctor
        return cmd_doctor()
    elif args.subcommand == "info":
        from neuralops.cli.info import cmd_info
        return cmd_info()

    return 0



if __name__ == "__main__":
    sys.exit(main())
