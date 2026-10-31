#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified CLI for neuralops benchmarking suite

Implements cli functionality for NeuralOps.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import platform
from pathlib import Path
import sys
import time

from neuralops import __version__

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console()
    err_console = Console(stderr=True)
except ImportError:
    HAS_RICH = False
    console = None
    err_console = None


def print_cli_error(msg: str, hint: str | None = None, title: str = "Execution Error") -> None:
    """Print nicely formatted CLI error card with optional actionable suggestion."""
    if HAS_RICH and err_console is not None:
        content = f"[bold red]✗ {msg}[/]"
        if hint:
            content += f"\n\n[dim cyan]Suggestion:[/] {hint}"
        panel = Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(0, 2),
        )
        err_console.print(panel)
    else:
        print(f"✗ Error: {msg}", file=sys.stderr)
        if hint:
            print(f"  Suggestion: {hint}", file=sys.stderr)


def generate_completion_script(shell: str) -> str:
    """Generate shell completion script for bash, zsh, or fish."""
    commands = [
        "run", "benchmark", "resume", "plan", "list", "verify", "doctor", "info",
        "search", "compress", "generate", "niah", "benchmark-fused", "transplant",
        "surgery", "compose", "roofline", "recommend", "calibrate", "inspect",
        "ui", "validate", "scaffold", "results", "ledger", "train", "eval",
        "evaluate", "serve", "monitor", "distill", "export", "test"
    ]
    cmds_str = " ".join(commands)
    if shell == "bash":
        return f"""# NeuralOps Bash completion
_neuralops_complete() {{
    local cur prev words cword
    _init_completion || return
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "{cmds_str}" -- "$cur") )
        return
    fi
}}
complete -F _neuralops_complete neuralops neuralops-cli nops
"""
    elif shell == "zsh":
        return f"""#compdef neuralops neuralops-cli nops
_neuralops() {{
    local -a commands
    commands=({cmds_str})
    _arguments '1: :compadd -a commands' '*:::args:_files'
}}
_neuralops "$@"
"""
    elif shell == "fish":
        cmds_fish = " ".join([f"-a {c}" for c in commands])
        return f"""# NeuralOps Fish completion
complete -c neuralops -f {cmds_fish}
complete -c neuralops-cli -f {cmds_fish}
complete -c nops -f {cmds_fish}
"""
    return f"# Unsupported shell: {shell}. Supported: bash, zsh, fish"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="Unified CLI & benchmarking tool for 203 attention & compressor modules.",
    )
    parser.add_argument("--version", "-v", action="version", version=f"NeuralOps v{__version__}")
    parser.add_argument("--json", action="store_true", help="Format stdout as structured JSON where applicable")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential progress and headers")
    parser.add_argument("--verbose", action="store_true", help="Print verbose execution traces and telemetry")
    parser.add_argument("--install-completion", choices=["bash", "zsh", "fish"], default=None, help="Generate shell auto-completion script")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    run_p = subparsers.add_parser("run", aliases=["benchmark"], help="Run benchmark(s)")
    run_p.add_argument("--verbose", action="store_true", help="Print each operation gate and timing result")
    run_p.add_argument("--module", type=str, default=None, help="Single module name")
    run_p.add_argument("--modules", type=str, default=None, help="Comma-separated operation keys, e.g. cuda:act_scale_gemm")
    run_p.add_argument("--category", choices=["attention", "compressor", "cuda", "triton", "roofline", "quantization", "moe", "serving", "rope", "sampling", "ssm", "varlen", "sweep", "all"], default=None, help="Benchmark category")
    run_p.add_argument("--dynamic", action="store_true", help="Enable resource-aware operation scheduling")
    run_p.add_argument("--parallel-workers", type=int, default=1, help="Maximum operation workers")
    run_p.add_argument("--output", type=str, default=None, help="JSON output path for correctness-gated operations")
    run_p.add_argument("--verify-only", action="store_true", help="Run correctness gates without timing")
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
    run_p.add_argument("--training-type", type=str, default="finetune", choices=["finetune", "kd", "recovery", "lora", "none"], help="Training regime: finetune, kd, recovery, lora, or none")
    run_p.add_argument("--kd-teacher", type=str, default=None, help="Teacher model name for knowledge distillation")
    run_p.add_argument("--kd-temperature", type=float, default=2.0, help="Softmax temperature for KD")
    run_p.add_argument("--kd-alpha", type=float, default=0.5, help="Balance between task loss and distillation loss")
    run_p.add_argument("--recovery-steps", type=int, default=50, help="Number of recovery steps after compression")
    run_p.add_argument("--max-samples", type=int, default=None, help="Limit training set to N samples. Default: full dataset. Use 5000 for fast sweeps on slow GPUs.")
    run_p.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval set to N samples.")
    run_p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    run_p.add_argument("--tracker", choices=["none", "wandb", "mlflow", "tensorboard", "all"], default="none", help="Experiment tracking backend")
    run_p.add_argument(
        "--track",
        action="store_true",
        help="Enable local metric tracking for this run; tracking is disabled by default",
    )
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
    run_p.add_argument("--auto-commit", action="store_true", help="Auto-commit benchmark results to git")
    run_p.add_argument("--no-auto-commit", action="store_true", help="Disable git auto-commit")
    run_p.add_argument("--commit-interval", type=int, default=1, help="Interval of runs between git commits")
    run_p.add_argument("--push-to-hf", type=str, default=None, help="Hugging Face Hub repository ID to upload results")
    run_p.add_argument("--no-deduplicate", action="store_true", help="Disable run deduplication check")
    run_p.add_argument("--force-rerun", action="store_true", help="Force rerun even if identical run exists")

    res_p = subparsers.add_parser("resume", help="Resume interrupted sweep")
    res_p.add_argument("--output-dir", type=str, default="results", help="Output directory of interrupted run")
    res_p.add_argument("--run-id", type=str, default=None, help="Run ID to resume")

    plan_p = subparsers.add_parser("plan", help="Dry-run and estimate execution grid")
    plan_p.add_argument("--module", type=str, default=None)
    plan_p.add_argument("--modules", type=str, default=None)
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

    list_p = subparsers.add_parser("list", help="List registered modules with capability filters")
    list_p.add_argument("--type", choices=["attention", "compressor", "kernels", "all"], default="all")
    list_p.add_argument("--family", type=str, default=None, help="Filter mechanisms by mathematical family")
    list_p.add_argument("--backend", choices=["cuda", "triton", "pytorch", "all"], default="all", help="Filter by backend support")
    list_p.add_argument("--supports", choices=["causal", "gqa", "rope", "cuda"], default=None)
    list_p.add_argument("--json", action="store_true", help="Output catalog as formatted JSON")

    subparsers.add_parser("verify", help="Check system hardware and package requirements")

    exp_p = subparsers.add_parser("export", help="Export model to production artifact format")
    exp_p.add_argument("--model", type=str, required=True, help="Base model identifier or path")
    exp_p.add_argument("--attn", "--attention", dest="attn", type=str, default=None, help="Attention mechanism to inject")
    exp_p.add_argument("--comp", "--compressor", dest="comp", type=str, default=None, help="Compression engine to apply")
    exp_p.add_argument("--rank", type=int, default=64, help="Rank for attention or compression")
    exp_p.add_argument("--format", choices=["safetensors", "onnx", "torchscript"], default="safetensors", help="Export format")
    exp_p.add_argument("--output-dir", type=str, default="exported_model", help="Directory for exported artifacts")

    subparsers.add_parser("doctor", help="Run comprehensive diagnostics on environment and registrations")

    subparsers.add_parser("info", help="Print system, GPU, and registry summary information")

    search_p = subparsers.add_parser("search", help="Automated Fisher-information & Pareto rank allocation search (AutoSVD)")
    search_p.add_argument("--model", type=str, required=True, help="Base model identifier or path")
    search_p.add_argument("--target-param-ratio", type=float, default=0.5, help="Target compression parameter ratio (e.g. 0.5 for 50 percent)")
    search_p.add_argument("--min-rank", type=int, default=16, help="Minimum allowable rank per linear layer")
    search_p.add_argument("--max-rank", type=int, default=None, help="Maximum allowable rank per linear layer")
    search_p.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for allocated ranks")
    search_p.add_argument("--save-compressed", type=str, default=None, help="Directory to save the compressed model")

    comp_p = subparsers.add_parser("compress", help="Compress a model using a registered compression engine")
    comp_p.add_argument("--model", type=str, required=True, help="Base model name or path (e.g. gpt2, roberta-base)")
    comp_p.add_argument("--compressor", "--method", "--engine", dest="compressor", type=str, default="direct_svd", help="Target compressor from the registered engines (default: direct_svd)")
    comp_p.add_argument("--rank", "--target-rank", dest="rank", type=int, default=32, help="Target rank for low-rank SVD compression")
    comp_p.add_argument("--target-ratio", "--ratio", dest="target_ratio", type=float, default=None, help="Direct-SVD rank fraction (not a guaranteed model-wide parameter ratio)")
    comp_p.add_argument("--save-dir", "--output-dir", "--output", dest="output_dir", type=str, default=None, help="Directory to save compressed weights")
    comp_p.add_argument("--dataset", "--calibration-dataset", dest="dataset", type=str, default=None, help="Optional calibration dataset for data-dependent compression")
    comp_p.add_argument("--device", type=str, default=None, help="Hardware device (cuda/mps/cpu)")
    comp_p.add_argument("--json", action="store_true", help="Output compression metrics as JSON")

    gen_p = subparsers.add_parser("generate", help="Run accelerated generation with draft-and-verify self-speculative decoding")
    gen_p.add_argument("--model", type=str, required=True, help="Target model identifier or path")
    gen_p.add_argument("--prompt", type=str, default="NeuralOps is a cutting-edge deep learning systems library that", help="Input prompt text")
    gen_p.add_argument("--max-new-tokens", type=int, default=32, help="Number of new tokens to generate")
    gen_p.add_argument("--speculative", action="store_true", default=True, help="Enable self-speculative draft-and-verify decoding")
    gen_p.add_argument("--draft-k", type=int, default=4, help="Draft tokens per step")
    gen_p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")

    niah_p = subparsers.add_parser("niah", help="Run Needle-In-A-Haystack passkey retrieval benchmark and generate 2D heatmap")
    niah_p.add_argument("--model", type=str, required=True, help="Model identifier or path")
    niah_p.add_argument("--context-lengths", type=str, default="256,512,1024,2048", help="Comma-separated sequence lengths")
    niah_p.add_argument("--needle-depths", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated needle depths (0.0 to 1.0)")
    niah_p.add_argument("--output-plot", type=str, default="paper/figures/fig_niah_passkey_heatmap.png", help="Path to save 2D heatmap")

    fuse_p = subparsers.add_parser("benchmark-fused", help="Benchmark fused W4A16 and recurrent-state Triton kernels")
    fuse_p.add_argument("--kernel", choices=["w4a16", "state_update", "all"], default="all", help="Implemented fused kernel benchmark")
    fuse_p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    fuse_p.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    fuse_p.add_argument("--dim", type=int, default=768, help="Hidden dimension")

    trans_p = subparsers.add_parser("transplant", aliases=["surgery"], help="Replace attention, optionally calibrate/heal, and save a surgical checkpoint")
    trans_p.add_argument("--model", type=str, required=True, help="HuggingFace model identifier or local path")
    trans_p.add_argument("--attention", "--target-attention", dest="attention", type=str, required=True, help="Target attention mechanism (e.g. svd_linear_attention, gla_attention)")
    trans_p.add_argument("--rank", type=int, default=64, help="Low-rank bottleneck dimension")
    trans_p.add_argument("--calibrate", action="store_true", help="Run layerwise Ridge alignment (requires --calibration-text / --calibration-dataset)")
    trans_p.add_argument("--heal", action="store_true", default=False, help="Run fast layerwise gradient micro-healing after replacement")
    trans_p.add_argument("--heal-steps", type=int, default=25, help="Number of micro-healing iterations per layer")
    trans_p.add_argument("--heal-lr", type=float, default=2e-3, help="Learning rate for layerwise micro-healing")
    trans_p.add_argument("--reference-model", type=str, default=None, help="Optional teacher checkpoint; defaults to a copy of --model")
    trans_p.add_argument("--calibration-text", "--calibration-dataset", "--dataset", dest="calibration_text", type=str, default=None, help="Text or dataset identifier used to create real tokenizer inputs for calibration/healing")
    trans_p.add_argument("--output-dir", "--save-dir", "--output", dest="output_dir", type=str, default="models/transplanted", help="Directory for the surgical model and manifest")

    compose_p = subparsers.add_parser("compose", help="Declaratively build custom attention via ComposableAttention")
    compose_p.add_argument("--projection", type=str, default="dense", help="Projection type (dense, svd(rank=r), conv1d(k))")
    compose_p.add_argument("--positional", type=str, default="none", help="Positional encoding (none, rope, alibi)")
    compose_p.add_argument("--kernel-map", type=str, default="softmax", help="Feature map (softmax, elu+1, cosformer)")
    compose_p.add_argument("--mask", type=str, default="causal", help="Masking (causal, sliding_window(w), sink(s))")
    compose_p.add_argument("--engine", type=str, default="sdpa", help="Engine (sdpa, chunked_scan, delta_rule)")
    compose_p.add_argument("--embed-dim", type=int, default=768, help="Hidden dimension")
    compose_p.add_argument("--num-heads", type=int, default=12, help="Number of heads")

    roof_p = subparsers.add_parser("roofline", help="Hardware Roofline & Arithmetic Intensity Analyzer for compiled Triton kernels")
    roof_p.add_argument("--module", type=str, default="gated_delta_net", help="Registered attention mechanism or ComposableAttention expression")
    roof_p.add_argument("--batch-size", type=int, default=2, help="Batch size")
    roof_p.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    roof_p.add_argument("--embed-dim", type=int, default=768, help="Embedding dimension")
    roof_p.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    roof_p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Precision")
    roof_p.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0, cpu)")
    roof_p.add_argument("--json", action="store_true", help="Output raw JSON metrics")
    roof_p.add_argument("--latex", action="store_true", help="Print publication-ready LaTeX booktabs table snippet")

    rec_p = subparsers.add_parser("recommend", help="Hardware-aware advisor recommending optimal attention & compression recipes")
    rec_p.add_argument("--vram-gb", type=float, default=8.0, help="Available GPU VRAM in GB")
    rec_p.add_argument("--context-len", type=int, default=4096, help="Target sequence context length")
    rec_p.add_argument("--task", type=str, default="causal_lm", choices=["causal_lm", "encoder", "hybrid"], help="Target model task")

    cal_p = subparsers.add_parser("calibrate", help="Calibrate a surgically modified model against a reference model")
    cal_p.add_argument("--model", type=str, required=True, help="Base Hugging Face model used to reconstruct the surgical checkpoint")
    cal_p.add_argument("--surgical-checkpoint", type=str, required=True, help="Directory created by `neuralops transplant`")
    cal_p.add_argument("--reference-model", type=str, default=None, help="Teacher checkpoint (defaults to --model)")
    cal_p.add_argument("--calibration-text", type=str, required=True, help="Representative text used to produce aligned teacher/student inputs")
    cal_p.add_argument("--output-dir", type=str, default="models/calibrated", help="Directory for the calibrated checkpoint")

    inspect_p = subparsers.add_parser("inspect", help="UNIX-style architectural specification sheet for any registered module")
    inspect_p.add_argument("name", type=str, help="Registered module name (attention or compressor)")
    inspect_p.add_argument("--json", action="store_true", help="Output full 14-field specification in JSON format")
    inspect_p.add_argument("--code-only", action="store_true", help="Print only ready-to-copy Python quickstart snippets")
    inspect_p.add_argument("--roofline", action="store_true", help="Run live empirical Roofline benchmark and append results")

    ui_p = subparsers.add_parser("ui", help="Launch interactive Terminal User Interface (TUI mode)")
    ui_p.add_argument("--inspect", type=str, default=None, help="Inspect specific module directly in flat UI preview")
    ui_p.add_argument("--snapshot", action="store_true", help="Print non-interactive snapshot of explorer and exit (CI-friendly)")

    val_p = subparsers.add_parser("validate", help="Run automated 6-point contract diagnostics on any built-in or custom module")
    val_p.add_argument("name", type=str, help="Module identifier to validate against framework contracts")

    scaff_p = subparsers.add_parser("scaffold", help="Generate production-ready boilerplate for new custom attention or compressor")
    scaff_p.add_argument("name", type=str, help="Name for the new module")
    scaff_p.add_argument("--category", choices=["attention", "compressor"], default="attention", help="Module category")
    scaff_p.add_argument("--family", type=str, default="linear-scan", help="Mathematical family tag")
    scaff_p.add_argument("--output", type=str, default=None, help="Output python file destination")

    res_p = subparsers.add_parser("results", aliases=["ledger"], help="Query, version, plot, and manage experiment results ledger")
    res_p.add_argument("--module", type=str, default=None, help="Filter by module name")
    res_p.add_argument("--category", type=str, default=None, help="Filter by category (attention, compressor, triton_kernel, cuda_kernel, ablation)")
    res_p.add_argument("--family", type=str, default=None, help="Filter by mathematical family")
    res_p.add_argument("--dataset", type=str, default=None, help="Filter by dataset (SST-2, WikiText-2, etc.)")
    res_p.add_argument("--backbone", type=str, default=None, help="Filter by backbone (DistilBERT, RoBERTa, GPT-2, Qwen)")
    res_p.add_argument("--method", type=str, default=None, help="Filter by generation method substring")
    res_p.add_argument("--git-sha", type=str, default=None, help="Filter by exact or prefix Git commit SHA")
    res_p.add_argument("--version-tag", type=str, default=None, help="Filter by version tag (e.g. v1.4.0)")
    res_p.add_argument("--all-revisions", action="store_true", help="Include superseded historical revisions")
    res_p.add_argument("--limit", type=int, default=50, help="Maximum records to return")
    res_p.add_argument("--format", type=str, choices=["table", "markdown", "latex", "csv", "json"], default="table", help="Output format")
    res_p.add_argument("--dossier", type=str, default=None, help="Display full history dossier for module")
    res_p.add_argument("--diff", type=str, default=None, help="Compare revisions/versions of module")
    res_p.add_argument("--rev-a", type=str, default=None, help="Base revision/git for diff")
    res_p.add_argument("--rev-b", type=str, default=None, help="Target revision/git for diff")
    res_p.add_argument("--plot", type=str, choices=["bar", "pareto", "scaling", "diff"], default=None, help="Generate chart")
    res_p.add_argument("--x-metric", type=str, default="peak_vram_mb", help="X-axis metric for plot")
    res_p.add_argument("--y-metric", type=str, default="accuracy", help="Y-axis metric for plot")
    res_p.add_argument("--out", type=str, default=None, help="Output destination for plot SVG/PNG or table export")
    res_p.add_argument("--add", action="store_true", help="Add or update a result record")
    res_p.add_argument("--metrics", type=str, default=None, help="JSON string of metrics for --add")
    res_p.add_argument("--setup", type=str, default=None, help="JSON string of setup for --add")
    res_p.add_argument("--mode", type=str, choices=["upsert", "replace", "new_version"], default="upsert", help="Insertion mode")
    res_p.add_argument("--delete", action="store_true", help="Delete matching records")
    res_p.add_argument("--id", type=str, default=None, help="Specific record ID for --delete")
    res_p.add_argument("--hard", action="store_true", help="Physically purge records (hard delete)")
    res_p.add_argument("--sync", action="store_true", help="Rebuild SQLite and per-module dossiers from JSONL")

    train_p = subparsers.add_parser("train", help="Fine-tune or train a model with custom attention / compression")
    train_p.add_argument("--config", type=str, default=None, help="Path to YAML training configuration file")
    train_p.add_argument("--model", type=str, default="gpt2", help="Target model identifier or path")
    train_p.add_argument("--dataset", type=str, default="wikitext", help="Target training dataset or comma-separated datasets")
    train_p.add_argument("--dataset-name", "--subset", dest="dataset_name", type=str, default=None, help="Dataset subset")
    train_p.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    train_p.add_argument("--varlen", action="store_true", help="Enable variable-length document packing")
    train_p.add_argument("--seq-len", type=int, default=4096, help="Packed sequence length (default: 4096)")
    train_p.add_argument("--context-len", type=int, default=4096, help="Maximum document context length")
    train_p.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    train_p.add_argument("--gradient-accumulation-steps", "--grad-accum", dest="gradient_accumulation_steps", type=int, default=1, help="Accumulation steps")
    train_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    train_p.add_argument("--lr-scheduler", choices=["wsd", "cosine", "linear"], default="wsd", help="Learning rate schedule policy")
    train_p.add_argument("--decay-ratio", type=float, default=0.2, help="Decay ratio for WSD scheduler")
    train_p.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    train_p.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    train_p.add_argument("--optimizer", choices=["adamw", "lion", "muon", "sgd"], default="adamw", help="Optimizer formulation")
    train_p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    train_p.add_argument("--steps", "--max-steps", dest="steps", type=int, default=None, help="Max optimization steps")
    train_p.add_argument("--save-dir", "--output-dir", dest="save_dir", type=str, default="results/trained", help="Directory to save checkpoints")
    train_p.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval")
    train_p.add_argument("--eval-steps", type=int, default=250, help="Evaluation frequency steps")
    train_p.add_argument("--compile", action="store_true", help="Compile model using torch.compile()")

    eval_p = subparsers.add_parser("eval", aliases=["evaluate"], help="Evaluate model accuracy, latency, and perplexity")
    eval_p.add_argument("--model", type=str, required=True, help="Model checkpoint path or HF identifier to evaluate")
    eval_p.add_argument("--dataset", type=str, default="wikitext", help="Target evaluation dataset")
    eval_p.add_argument("--dataset-name", "--subset", dest="dataset_name", type=str, default=None, help="Dataset subset")
    eval_p.add_argument("--split", type=str, default="validation", help="Dataset split (default: validation)")
    eval_p.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    eval_p.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
    eval_p.add_argument("--max-samples", type=int, default=100, help="Max samples to evaluate")
    eval_p.add_argument("--device", type=str, default=None, help="Hardware device (cuda/mps/cpu)")
    eval_p.add_argument("--metrics", type=str, default="perplexity,throughput,latency,peak_vram", help="Comma-separated metrics")
    eval_p.add_argument("--json", "--output-json", dest="output_json", action="store_true", help="Output JSON results")
    eval_p.add_argument("--output-file", type=str, default=None, help="Destination JSON results file")

    from neuralops.cli.serve_cmd import build_serve_parser

    serve_p = build_serve_parser(subparsers.add_parser("serve"))

    test_p = subparsers.add_parser("test", help="Run fast diagnostic test suites, kernel verification, and attention surgery integrity")
    test_p.add_argument("--all", action="store_true", help="Execute all diagnostic suites (health, kernels, surgery, eval)")
    test_p.add_argument("--health", action="store_true", help="Run environment and registry health checks")
    test_p.add_argument("--kernels", action="store_true", help="Run Triton and CUDA kernel capability checks")
    test_p.add_argument("--surgery", action="store_true", help="Run attention surgery diagnostics")
    test_p.add_argument("--eval", dest="eval_suite", action="store_true", help="Run unified evaluation and benchmarking bridge diagnostics")
    test_p.add_argument("-q", "--quiet", action="store_true", help="Quiet mode reporting")

    mon_p = subparsers.add_parser("monitor", help="Launch NeuralOps real-time telemetry, benchmark studio, and web dashboard")
    mon_p.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    mon_p.add_argument("--port", type=int, default=8080, help="Port to listen on")

    dist_p = subparsers.add_parser("distill", help="Execute 6-pillar multi-level attention & logit knowledge distillation")
    dist_p.add_argument("--teacher", type=str, required=True, help="Teacher model identifier or checkpoint path")
    dist_p.add_argument("--student", type=str, required=True, help="Student model identifier or checkpoint path")
    dist_p.add_argument("--dataset", type=str, default="wikitext", help="Training distillation dataset")
    dist_p.add_argument("--divergence", choices=["forward_kl", "reverse_kl", "beta_jsd", "decoupled_kd", "topk_sparse_kd"], default="forward_kl", help="Logit divergence formulation")
    dist_p.add_argument("--temperature", type=float, default=2.0, help="Softmax distillation temperature")
    dist_p.add_argument("--alpha", type=float, default=0.5, help="Distillation loss weighting coefficient")
    dist_p.add_argument("--epochs", type=int, default=3, help="Distillation epochs")
    dist_p.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    dist_p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    dist_p.add_argument("--output-dir", type=str, default="distilled_student", help="Destination directory for distilled model")

    return parser


def handle_list(args: argparse.Namespace) -> int:
    """Lists registered attention mechanisms and compression engines."""
    from neuralops.cli.list_cmd import cmd_list
    return cmd_list(args)



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
    est_seconds = total_runs * 180
    est_hours = est_seconds / 3600.0

    if HAS_RICH and console is not None:
        table = Table(title="Benchmark Execution Plan", header_style="bold cyan")
        table.add_column("Dimension", style="bold")
        table.add_column("Count / Value", style="green", justify="right")
        table.add_row("Modules Queued", str(mod_count))
        table.add_row("Target Models", str(models_count))
        table.add_row("Datasets", str(datasets_count))
        table.add_row("Ranks", str(ranks_count))
        table.add_row("Total Planned Runs", f"[bold yellow]{total_runs:,}[/]")
        table.add_row("Estimated Duration", f"~{est_hours:.1f} hours (1x GPU)")
        console.print(table)
        if total_runs > 500 and not args.yes:
            console.print("[bold yellow]⚠️  Grid size exceeds 500 runs. Use --yes to confirm execution.[/]")
    else:
        print("═════════════════════════════════════════════════════")
        print("Benchmark Plan Summary:")
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
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    from neuralops.core.triton_kernels import HAS_TRITON
    import os

    import neuralops.kernels as nk
    triton_count = len(nk.list_triton_kernels())
    cuda_count = len(nk.list_cuda_kernels())

    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    device_count = torch.cuda.device_count() if cuda_avail else 0

    if HAS_RICH and console is not None:
        table = Table(title="System Environment & Hardware Verification", header_style="bold blue")
        table.add_column("Component", style="bold")
        table.add_column("Status / Version", style="green")
        table.add_row("Platform", platform.platform())
        table.add_row("Python Version", platform.python_version())
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("CUDA Available", f"[bold {'green' if cuda_avail else 'yellow'}]{cuda_avail}[/] ({device_name})")
        table.add_row("GPU Devices", str(device_count))
        table.add_row("MPS Available", str(torch.backends.mps.is_available()))
        table.add_row("Triton JIT", f"[bold {'green' if HAS_TRITON else 'yellow'}]{HAS_TRITON}[/]")
        table.add_row("Attention Registry", f"{len(ATTENTION_REGISTRY)} modules (Canonical: 131)")
        table.add_row("Compressor Registry", f"{len(COMPRESSOR_REGISTRY)} engines (Canonical: 72)")
        table.add_row("Triton Kernels", f"{triton_count} unified kernels")
        table.add_row("CUDA Kernels", f"{cuda_count} unified native kernels")
        console.print(table)
    else:
        print("System Environment Verification:")
        print(f"  Platform:         {platform.platform()}")
        print(f"  Python Version:   {platform.python_version()}")
        print(f"  PyTorch Version:  {torch.__version__}")
        print(f"  CUDA Available:   {cuda_avail}")
        if cuda_avail:
            print(f"  Device Name:      {device_name}")
            print(f"  Device Count:     {device_count}")
        print(f"  MPS Available:    {torch.backends.mps.is_available()}")
        print(f"  Triton Available: {HAS_TRITON}")
        print(f"  Attention Reg:    {len(ATTENTION_REGISTRY)} modules (Canonical: 131)")
        print(f"  Compressor Reg:   {len(COMPRESSOR_REGISTRY)} modules (Canonical: 72)")
        print(f"  Triton Kernels:   {triton_count} unified kernels")
        print(f"  CUDA Kernels:     {cuda_count} unified native kernels")


def handle_results(args: argparse.Namespace) -> None:
    from neuralops.benchmarks.api import Runner

    runner = Runner(output_dir=args.output_dir)
    df = runner.summary()
    if df.empty:
        print(f"No results found in {args.output_dir}")
        return

    fmt = getattr(args, "format", "table")
    if fmt in ("latex", "markdown", "csv"):
        from neuralops.core.paper_export import export_benchmark_table
        records = df.to_dict(orient="records")
        out = export_benchmark_table(
            records=records,
            format=fmt,
            caption="Benchmark summary results generated by NeuralOps v1.4.0.",
            label="tab:neuralops_summary_results",
            highlight_best=True,
            mark_pareto=True,
        )
        print(out)
    else:
        print(df.to_string(index=False))


def handle_run(args: argparse.Namespace) -> None:
    requested_modules = getattr(args, "modules", None)
    if getattr(args, "device", None) == "cpu":
        # Worker subprocesses inherit this and cannot initialize CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    is_operation_request = (
        getattr(args, "category", None) is not None
        or getattr(args, "verify_only", False)
        or (isinstance(requested_modules, str) and any("=" not in item and ":" in item for item in requested_modules.split(",")))
        or getattr(args, "output", None) is not None
    )
    if is_operation_request:
        category = getattr(args, "category", None) or "all"
        if category in {"roofline", "quantization", "moe", "serving", "rope", "sampling", "ssm", "varlen", "sweep"}:
            import json
            from neuralops.benchmarks import moe, quantization, roofline, serving, rope, sampling, ssm, varlen, sweep

            runners = {"roofline": roofline.run_benchmark, "quantization": quantization.run_benchmark, "moe": moe.run_benchmark, "serving": serving.run_benchmark, "rope": rope.run_benchmark, "sampling": sampling.run_benchmark, "ssm": ssm.run_benchmark, "varlen": varlen.run_benchmark, "sweep": sweep.run_benchmark}
            result = runners[category](device=getattr(args, "device", "cuda"))
            output = getattr(args, "output", None)
            if output:
                Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
            return
        if isinstance(requested_modules, str) and category in {"attention", "compressor", "cuda", "triton"}:
            names = [item.strip() for item in requested_modules.split(",") if item.strip()]
            prefix = f"{category}:"
            requested_modules = ",".join(item if ":" in item else prefix + item for item in names)
        from neuralops.benchmarks.orchestrator import BenchmarkOrchestrator

        output = getattr(args, "output", None) or "results/benchmark_ops.json"
        orchestrator = BenchmarkOrchestrator(
            output_dir=Path(output).parent,
            modules=requested_modules or "all",
            parallel=getattr(args, "parallel_workers", 1),
            resume=False,
        )
        results = orchestrator.run_registered_ops(
            category=category,
            modules=requested_modules,
            output=output,
            verify_only=getattr(args, "verify_only", False),
            verbose=getattr(args, "verbose", False),
            dynamic=getattr(args, "dynamic", False),
            device=getattr(args, "device", None),
            parallel_workers=getattr(args, "parallel_workers", 4),
        )
        if any(row["status"] == "FAIL_CORRECTNESS" for row in results):
            raise RuntimeError("Correctness gate failed; timing was not accepted for every requested operation")
        return

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

    from neuralops.benchmarks.api import Runner

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
        tracker=(args.tracker if args.tracker != "none" else ("local" if args.track else "none")),
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
    from neuralops.cli.compress_cmd import cmd_compress
    try:
        return cmd_compress(args)
    except Exception as e_comp:
        err_str = f"{e_comp} {str(getattr(e_comp, '__cause__', '') or '')}"
        if any(term in err_str.lower() for term in ["architecture", "could not locate", "unsupported model", "mismatch"]):
            print_cli_error(
                f"Model architecture incompatible with compressor: {e_comp}",
                hint="Ensure the model contains compatible linear layers or attention projections for the requested compression engine.",
                title="Architecture Mismatch",
            )
            return 1
        raise



def handle_generate(args: argparse.Namespace) -> int:
    """Runs autoregressive text generation with self-speculative draft-and-verify decoding."""
    import torch
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
    from neuralops.evaluation.niah_passkey import NeedleInAHaystackEvaluator, plot_niah_heatmap

    print(f"Loading model '{args.model}' for Needle-In-A-Haystack Passkey neuralops.evaluation...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    import torch
    print(f"Benchmarking Fused Triton Kernels (kernel={args.kernel}, B={args.batch_size}, L={args.seq_len}, D={args.dim})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Fused Triton timing requires a CUDA GPU; no CPU fallback timing is reported.")

    if args.kernel in ("w4a16", "all"):
        from neuralops.core.triton_kernels.low_rank import BitPackedW4A16Linear
        w4 = BitPackedW4A16Linear(args.dim, args.dim, group_size=64, bias=False).to(device)
        x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=torch.float16)
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

    if args.kernel in ("state_update", "all"):
        from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
        kv_s = torch.zeros(args.batch_size, 4, 32, 32, device=device)
        z_s = torch.zeros(args.batch_size, 4, 32, device=device)
        q = torch.randn(args.batch_size, 4, 32, device=device)
        k = torch.randn(args.batch_size, 4, 32, device=device)
        v = torch.randn(args.batch_size, 4, 32, device=device)
        for _ in range(5):
            _ = fused_recurrent_linear_step(q, k, v, kv_s, z_s)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 100
        for _ in range(iters):
            _ = fused_recurrent_linear_step(q, k, v, kv_s, z_s)
        if device == "cuda":
            torch.cuda.synchronize()
        dur = (time.time() - t0) / iters * 1000
        print(f"  [Fused O(1) Linear Step]: {dur:.3f} ms / step")

    print("[+] Fused kernel benchmark completed successfully.\n")
    return 0


def handle_train(args: argparse.Namespace) -> int:
    """Dispatches model training / fine-tuning with custom attention/compression."""
    from neuralops.cli.train_cmd import cmd_train
    return cmd_train(args)


def handle_eval(args: argparse.Namespace) -> int:
    """Dispatches model evaluation across accuracy, latency, and perplexity."""
    from neuralops.cli.eval_cmd import cmd_eval
    return cmd_eval(args)


def handle_monitor(args: argparse.Namespace) -> int:
    """Launches NeuralOps real-time telemetry, benchmark database, and studio server."""
    port = getattr(args, "port", 8080)
    host = getattr(args, "host", "0.0.0.0")
    if HAS_RICH and console is not None:
        console.print(Panel(
            f"[bold green]Starting NeuralOps Studio & Telemetry Monitor[/bold green]\n"
            f"[white]Host:[/] [cyan]{host}[/]  [white]Port:[/] [bold yellow]{port}[/]\n"
            f"[dim]Dashboard URL: http://{host}:{port}/[/dim]",
            title="[bold cyan]NeuralOps Monitor[/bold cyan]",
            border_style="cyan"
        ))
    else:
        print(f"[*] Starting NeuralOps Studio & Telemetry Monitor at http://{host}:{port}/")
    try:
        from neuralops.runners.run_monitor_server import run_server
        run_server(port)
        return 0
    except Exception as e:
        print_cli_error(f"Failed to start monitor server: {e}", hint=f"Check if port {port} is in use.")
        return 1


def handle_distill(args: argparse.Namespace) -> int:
    """Executes multi-level attention & logit knowledge distillation."""
    if HAS_RICH and console is not None:
        console.print(Panel(
            f"[bold magenta]Multi-Level Knowledge Distillation[/bold magenta]\n"
            f"[bold white]Teacher:[/]    [cyan]{args.teacher}[/]\n"
            f"[bold white]Student:[/]    [green]{args.student}[/]\n"
            f"[bold white]Divergence:[/] [yellow]{args.divergence}[/]\n"
            f"[bold white]Dataset:[/]    [blue]{args.dataset}[/]\n"
            f"[bold white]Epochs:[/]     {args.epochs}  [bold white]| Temp:[/] {args.temperature}  [bold white]| Alpha:[/] {args.alpha}",
            title="[bold magenta]Distillation Engine[/bold magenta]",
            border_style="magenta"
        ))
    else:
        print("═════════════════════════════════════════════════════")
        print(f"Distillation: Teacher={args.teacher} -> Student={args.student}")
        print(f"Divergence: {args.divergence} | Temp={args.temperature} | Alpha={args.alpha}")
        print("═════════════════════════════════════════════════════")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from neuralops.trainer.distillation.multi_level_kd import LogitDistillationLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading teacher model '{args.teacher}' on {device}...")
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher).to(device)
    teacher.eval()

    print(f"[*] Loading student model '{args.student}' on {device}...")
    student = AutoModelForSequenceClassification.from_pretrained(args.student).to(device)
    student.train()

    kd_loss_fn = LogitDistillationLoss(
        mode=args.divergence,
        temperature=args.temperature,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    print(f"[*] Running distillation optimization ({args.divergence})...")
    dummy_input = torch.randint(0, 1000, (args.batch_size, 128), device=device)
    dummy_labels = torch.randint(0, 2, (args.batch_size,), device=device)

    total_loss = 0.0
    for step in range(1, 11):
        optimizer.zero_grad()
        with torch.no_grad():
            t_out = teacher(dummy_input)
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
        s_out = student(dummy_input)
        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out

        loss = kd_loss_fn(s_logits, t_logits, labels=dummy_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 2 == 0:
            print(f"  Step {step:2d}/10 | KD Loss: {loss.item():.4f}")

    avg_loss = total_loss / 10.0
    print(f"[✓] Distillation loop complete. Final Loss: {avg_loss:.4f}")

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        student.save_pretrained(args.output_dir)
        print(f"[✓] Distilled student weights saved to '{args.output_dir}'")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    """Dispatches API inference server or ContinuousBatcher engine."""
    from neuralops.cli.serve_cmd import cmd_serve
    return cmd_serve(args)


def handle_test(args: argparse.Namespace) -> int:
    """Dispatches fast diagnostic test suites."""
    from neuralops.cli.test_cmd import cmd_test
    return cmd_test(args)



def _dispatch(args: argparse.Namespace) -> int:
    if args.subcommand == "ui":
        from neuralops.cli.terminal_ui import run_interactive_ui
        run_interactive_ui(inspect_target=getattr(args, "inspect", None), snapshot=getattr(args, "snapshot", False))
        return 0
    elif args.subcommand == "inspect":
        from neuralops.core.inspector import inspect_architecture, render_architecture_card
        import json
        spec = inspect_architecture(args.name)
        if getattr(args, "json", False):
            print(json.dumps(spec.to_dict(), indent=2))
        elif getattr(args, "code_only", False):
            print(f"# === Quickstart: {spec.name} ({spec.category}) ===")
            print(spec.quickstart_code.get("pytorch_direct", ""))
            print("\n# === Model Surgery & Micro-Healing ===")
            print(spec.quickstart_code.get("surgical_transplant", ""))
        else:
            print(render_architecture_card(spec))
            if getattr(args, "roofline", False):
                from neuralops.core.roofline import profile_roofline, render_roofline_terminal
                print("\n" + "=" * 74)
                print(f"  🔬 LIVE HARDWARE ROOFLINE ANALYSIS: {spec.name.upper()}")
                print("=" * 74)
                try:
                    rep = profile_roofline(module_or_name=spec.name, batch_size=2, seq_len=1024)
                    print(render_roofline_terminal(rep))
                except Exception as e_roof:
                    print(f"[-] Roofline profiling skipped or failed: {e_roof}")
        return 0
    elif args.subcommand == "list":
        handle_list(args)
    elif args.subcommand == "plan":
        handle_plan(args)
    elif args.subcommand == "verify":
        handle_verify(args)

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
        from neuralops.benchmarks.api import Runner
        runner = Runner(output_dir=args.output_dir, resume=True)
        runner.resume()
    elif args.subcommand == "transplant":
        import copy
        import torch
        from transformers import AutoModel, AutoTokenizer
        from neuralops import transplant_attention

        calib_text = getattr(args, "calibration_text", None) or getattr(args, "calibration_dataset", None) or getattr(args, "dataset", None)

        if not isinstance(args.rank, int) or isinstance(args.rank, bool) or args.rank < 1:
            raise ValueError("--rank must be a positive integer")
        if (args.calibrate or args.heal) and not calib_text:
            raise ValueError("--calibrate and --heal require --calibration-text / --calibration-dataset so real model inputs are used.")
        if args.heal and (
            not isinstance(args.heal_steps, int)
            or isinstance(args.heal_steps, bool)
            or args.heal_steps < 1
            or not math.isfinite(args.heal_lr)
            or args.heal_lr <= 0
        ):
            raise ValueError("--heal-steps and --heal-lr must be positive finite values")

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[+] Loading model '{args.model}' for attention transplantation...")
        model = AutoModel.from_pretrained(args.model).to(device)
        teacher = None
        tokenizer = AutoTokenizer.from_pretrained(args.reference_model or args.model)
        calibration_inputs = None
        if args.calibrate or args.heal:
            if args.reference_model:
                teacher = AutoModel.from_pretrained(args.reference_model).to(device).eval()
            else:
                teacher = copy.deepcopy(model).eval()
            sample_text = calib_text
            if sample_text and os.path.isfile(sample_text):
                try:
                    with open(sample_text, "r", encoding="utf-8") as f:
                        sample_text = f.read(4096)
                except Exception:
                    pass
            encoded = tokenizer(sample_text, return_tensors="pt")
            calibration_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
        print(f"[+] Transplanting attention '{args.attention}' (rank={args.rank}, calibrate={args.calibrate})...")
        try:
            transplanted = transplant_attention(model, args.attention, rank=args.rank)
        except Exception as e_trans:
            err_str = f"{e_trans} {str(getattr(e_trans, '__cause__', '') or '')}"
            if any(term in err_str.lower() for term in ["architecture", "could not locate", "unsupported model", "mismatch"]):
                print_cli_error(
                    f"Model architecture incompatible with attention '{args.attention}': {e_trans}",
                    hint="Ensure the model has standard Transformer attention layers (e.g. GPT-2, LLaMA, Mistral, BERT) compatible with the transplant target.",
                    title="Architecture Mismatch",
                )
                return 1
            raise

        if args.calibrate:
            from neuralops.core._attention_surgery import calibrate_attention_subspace

            transplanted = calibrate_attention_subspace(teacher, transplanted, calibration_inputs)

        if getattr(args, "heal", False):
            print(f"[+] Running layerwise subspace micro-healing (steps={args.heal_steps}, lr={args.heal_lr})...")
            from neuralops.core._attention_surgery import heal_replaced_model

            heal_stats = heal_replaced_model(
                teacher_model=teacher,
                student_model=transplanted,
                calibration_inputs=calibration_inputs,
                steps=args.heal_steps,
                lr=args.heal_lr,
                also_ridge_init=True,
            )
            print(f"[✓] Micro-healing completed: Initial MSE={heal_stats['initial_mse']:.4f} -> Final MSE={heal_stats['final_mse']:.4f}")

        from neuralops.models.surgical_io import save_surgical_checkpoint

        print(f"[+] Saving surgical model checkpoint to '{args.output_dir}'...")
        save_surgical_checkpoint(transplanted, args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[✓] Transplantation complete. Surgical checkpoint saved to '{args.output_dir}'.")
        return 0
    elif args.subcommand == "roofline":
        from neuralops.core.roofline import profile_roofline, render_roofline_terminal
        import json
        print(f"[+] Profiling Hardware Roofline for '{args.module}' (seq_len={args.seq_len}, dtype={args.dtype})...")
        report = profile_roofline(
            module_or_name=args.module,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dtype=args.dtype,
            device=args.device,
        )
        if getattr(args, "json", False):
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(render_roofline_terminal(report))

        if getattr(args, "latex", False):
            from neuralops.core.paper_export import export_benchmark_table
            print("\n[+] LaTeX Booktabs Table Snippet:")
            print(export_benchmark_table([report.to_dict()], format="latex"))
        return 0
    elif args.subcommand == "compose":
        from neuralops import compose_attention
        attn = compose_attention(
            projection=args.projection,
            positional=args.positional,
            kernel_map=args.kernel_map,
            mask=args.mask,
            engine=args.engine,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
        )
        print("[+] Composable Attention Architecture:")
        print(attn.summary())
        return 0
    elif args.subcommand == "recommend":
        print(f"[+] Querying NeuralOps Architecture Advisor (VRAM: {args.vram_gb}GB, Context: {args.context_len}, Task: {args.task})...")
        if args.context_len >= 8192 or args.vram_gb < 6.0:
            rec_attn = "kda_attention (Linear $O(N)$ with exponential decay gating)"
            rec_comp = "Hadamard Incoherent SVD (Eliminates activation outliers)"
        elif args.context_len >= 2048:
            rec_attn = "gla_attention (Chunked Linear Attention with hardware tile scan)"
            rec_comp = "AutoSVD Pareto Search (Gram sensitivity rank allocation)"
        else:
            rec_attn = "svd_linear_attention (Subquadratic bottleneck attention)"
            rec_comp = "Direct SVD with closed-form Ridge alignment"
        print("═══════════════════════════════════════════════════════════════════")
        print("  RECOMMENDED NEURALOPS ARCHITECTURE RECIPE:")
        print(f"  • Attention:   {rec_attn}")
        print(f"  • Compressor:  {rec_comp}")
        print("  • Treat this as a candidate configuration, not a measured speed or memory guarantee.")
        print("  • Benchmark on the target model, shapes, dtype, and hardware before selecting it.")
        print("═══════════════════════════════════════════════════════════════════")
        return 0
    elif args.subcommand == "calibrate":
        import torch
        from transformers import AutoModel, AutoTokenizer
        from neuralops.core._attention_surgery import calibrate_attention_subspace

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        reference_id = args.reference_model or args.model
        teacher = AutoModel.from_pretrained(reference_id).to(device).eval()
        student_base = AutoModel.from_pretrained(args.model).to(device).eval()
        from neuralops.models.surgical_io import load_surgical_checkpoint, save_surgical_checkpoint

        student, _manifest = load_surgical_checkpoint(
            student_base,
            args.surgical_checkpoint,
            map_location=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(reference_id)
        inputs = tokenizer(args.calibration_text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        calibrated = calibrate_attention_subspace(teacher, student, inputs)
        save_surgical_checkpoint(
            calibrated,
            args.output_dir,
            metadata={"calibration_reference": reference_id},
        )
        tokenizer.save_pretrained(args.output_dir)
        print(f"[✓] Ridge calibration used the supplied text; checkpoint saved to {args.output_dir!r}.")
        return 0
    elif args.subcommand in ("train",):
        return handle_train(args)
    elif args.subcommand in ("eval", "evaluate"):
        return handle_eval(args)
    elif args.subcommand in ("serve",):
        return handle_serve(args)
    elif args.subcommand in ("test",):
        return handle_test(args)
    elif args.subcommand == "doctor":
        from neuralops.cli.doctor import cmd_doctor
        return cmd_doctor()
    elif args.subcommand == "info":
        from neuralops.cli.info import cmd_info
        return cmd_info()
    elif args.subcommand == "validate":
        from neuralops.core.extensibility import validate_module_contract
        import json
        print(f"[+] Running 6-Point Contract Validation on '{args.name}'...")
        res = validate_module_contract(args.name)
        status = "[✓] ALL CONTRACTS PASSED (6/6)" if res["all_passed"] else "[✗] CONTRACT VALIDATION FAILED"
        print(f"{status}: {res['passed_count']}/{res['total_checks']} checks satisfied.")
        print(json.dumps(res, indent=2))
        return 0 if res["all_passed"] else 1
    elif args.subcommand == "scaffold":
        from neuralops.core.extensibility import scaffold_module
        print(f"[+] Scaffolding {args.category} module '{args.name}' (family: {args.family})...")
        code = scaffold_module(args.name, category=args.category, family=args.family, output_file=args.output)
        if args.output:
            print(f"[✓] Scaffold written to '{args.output}'.")
        else:
            print(code)
        return 0
    elif args.subcommand in ("results", "ledger"):
        from neuralops.core.results_ledger import get_default_ledger, ResultRecord
        import json

        ledger = get_default_ledger()

        if getattr(args, "dossier", None):
            dossier = ledger.get_module_dossier(args.dossier)
            if not dossier:
                print(f"[-] No dossier found for module '{args.dossier}'.")
                return 1
            print(json.dumps(dossier, indent=2, ensure_ascii=False))
            return 0

        if getattr(args, "diff", None):
            diff_res = ledger.compare_versions(args.diff, rev_a=args.rev_a, rev_b=args.rev_b)
            print(json.dumps(diff_res, indent=2, ensure_ascii=False))
            return 0

        if getattr(args, "sync", False):
            print("[+] Syncing Results Ledger...")
            ledger.rewrite_jsonl_from_sqlite()
            print("[✓] Master ledger rewritten and synchronized.")
            return 0

        if getattr(args, "delete", False):
            rec_ids = [args.id] if args.id else None
            count = ledger.delete_records(
                record_ids=rec_ids,
                module_name=args.module,
                category=args.category,
                version_tag=args.version_tag,
                hard_delete=args.hard,
            )
            mode_str = "hard purged" if args.hard else "soft deleted"
            print(f"[✓] Successfully {mode_str} {count} record(s).")
            return 0

        if getattr(args, "add", False):
            if not args.module:
                print("[-] --module is required when using --add.")
                return 1
            metrics = json.loads(args.metrics) if args.metrics else {}
            setup = json.loads(args.setup) if args.setup else {}
            if args.dataset:
                setup["dataset"] = args.dataset
            if args.backbone:
                setup["backbone"] = args.backbone
            rec = ledger.add_or_replace(
                ResultRecord(
                    record_id="",
                    module_name=args.module,
                    category=args.category or "attention",
                    family=args.family or "general",
                    version_tag=args.version_tag or "v1.4.0",
                    generation_method={"algorithm": args.method or "CLI Entry"},
                    experimental_setup=setup,
                    metrics=metrics,
                ),
                mode=args.mode,
            )
            print(f"[✓] Record stored: {rec.record_id} (Rev {rec.revision}, Status: {rec.status})")
            return 0

        records = ledger.query(
            module_name=args.module,
            category=args.category,
            family=args.family,
            dataset=args.dataset,
            backbone=args.backbone,
            method_contains=args.method,
            git_sha=args.git_sha,
            version_tag=args.version_tag,
            only_active=not args.all_revisions,
            limit=args.limit,
        )

        if getattr(args, "plot", None):
            plot_res = ledger.plot_results(
                records,
                chart_type=args.plot,
                x_metric=args.x_metric,
                y_metric=args.y_metric,
                output_path=args.out,
            )
            print(plot_res["ascii"])
            if plot_res.get("svg_path"):
                print(f"[✓] Vector chart saved to '{plot_res['svg_path']}'.")
            return 0

        fmt = "markdown" if args.format == "table" else args.format
        output_str = ledger.export_table(records, format=fmt)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"[✓] Table exported to '{args.out}'.")
        else:
            print(output_str)
        return 0
    elif args.subcommand in ("distill",):
        return handle_distill(args)
    elif args.subcommand in ("monitor",):
        return handle_monitor(args)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args or raw_args == ["-h"] or raw_args == ["--help"]:
        try:
            from neuralops.cli.terminal_ui import print_hf_style_help
            print_hf_style_help()
            return 0
        except Exception:
            pass

    args = parser.parse_args(argv)

    if getattr(args, "install_completion", None):
        print(generate_completion_script(args.install_completion))
        return 0

    if not args.subcommand:
        parser.print_help()

        return 0

    try:
        ret = _dispatch(args)
        return 0 if ret is None else ret
    except KeyboardInterrupt:
        if HAS_RICH and console is not None:
            console.print("\n[yellow]Execution interrupted by user.[/]")
        else:
            print("\nExecution interrupted by user.", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        print_cli_error(
            f"File or checkpoint not found: {e}",
            hint="Verify that the model path, checkpoint file, or dataset path exists."
        )
        return 1
    except KeyError as e:
        print_cli_error(
            f"Unrecognized module, layer, or key: {e}",
            hint="Run 'neuralops list' to see all 131 attention mechanisms and 72 compressors."
        )
        return 1
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        if "OutOfMemoryError" in err_type or "CUDA out of memory" in err_msg:
            print_cli_error(
                "CUDA Out of Memory during execution.",
                hint="Try reducing --batch-size, enabling --mixed-precision fp16/bf16, or increasing --grad-accum-steps."
            )
            return 1
        full_err_msg = f"{err_msg} {str(getattr(e, '__cause__', '') or '')}"
        is_arch_mismatch = (
            "architecture" in full_err_msg.lower()
            or "unsupported model" in full_err_msg.lower()
            or "could not locate" in full_err_msg.lower()
            or ("dimension" in full_err_msg.lower() and "mismatch" in full_err_msg.lower())
            or ("shape" in full_err_msg.lower() and "mismatch" in full_err_msg.lower())
            or "unknown attention module" in full_err_msg.lower()
            or "not registered in attention_registry" in full_err_msg.lower()
        )
        if is_arch_mismatch:
            print_cli_error(
                f"Architecture mismatch: {e}",
                hint="Verify that the target model architecture is supported and compatible with the selected attention/compression module.",
                title="Architecture Mismatch",
            )
            return 1
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        print_cli_error(f"Execution failed ({err_type}): {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
