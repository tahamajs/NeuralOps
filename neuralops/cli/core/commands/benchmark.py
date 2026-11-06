#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Benchmark command family: run, plan, resume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuralops.cli.core.config_merge import load_effective_config
from neuralops.cli.core.devices import resolve_device
from neuralops.cli.core.errors import ExitCode, ExecutionError
from neuralops.cli.core.locks import create_run_directory, run_directory_lock
from neuralops.cli.core.output import CliReporter


def handle_plan(args: argparse.Namespace) -> int:
    """Preview benchmark execution plan without allocating GPU tensors or modifying disk."""
    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )
    cli_dict = {k: v for k, v in vars(args).items() if v is not None}
    config_path = getattr(args, "config", None)
    effective_cfg = load_effective_config(cli_dict, config_path=config_path)

    from neuralops.benchmarks.orchestrator import BenchmarkOrchestrator

    modules_arg = effective_cfg.get("modules") or effective_cfg.get("module") or "all"
    orch = BenchmarkOrchestrator(
        modules=modules_arg,
        delta=effective_cfg.get("all_delta", False),
        gated=effective_cfg.get("all_gated", False),
        svd=effective_cfg.get("all_svd", False),
        ssm=effective_cfg.get("all_ssm", False),
        linear=effective_cfg.get("all_linear", False),
        sparse=effective_cfg.get("all_sparse", False),
        hybrid=effective_cfg.get("all_hybrid", False),
        all_attn=effective_cfg.get("all_attn", False),
        all_comp=effective_cfg.get("all_comp", False),
        all_modules=effective_cfg.get("all_modules", False),
        fail_delta=effective_cfg.get("fail_delta", False),
        errors_only=effective_cfg.get("errors_only", False),
    )
    mod_count = len(orch.modules)
    models_raw = effective_cfg.get("models") or effective_cfg.get("model")
    datasets_raw = effective_cfg.get("datasets") or effective_cfg.get("dataset")
    ranks_raw = effective_cfg.get("ranks") or effective_cfg.get("rank")

    models_count = 10 if effective_cfg.get("all_models") else (len(models_raw.split(",")) if isinstance(models_raw, str) else (len(models_raw) if isinstance(models_raw, list) else 1))
    datasets_count = 22 if effective_cfg.get("all_datasets") else (len(datasets_raw.split(",")) if isinstance(datasets_raw, str) else (len(datasets_raw) if isinstance(datasets_raw, list) else 1))
    ranks_count = len(ranks_raw.split(",")) if isinstance(ranks_raw, str) else (len(ranks_raw) if isinstance(ranks_raw, list) else 1)

    total_runs = mod_count * models_count * datasets_count * ranks_count
    est_seconds = total_runs * 180
    est_hours = est_seconds / 3600.0

    plan_data = {
        "modules_queued": mod_count,
        "models_count": models_count,
        "datasets_count": datasets_count,
        "ranks_count": ranks_count,
        "total_planned_runs": total_runs,
        "estimated_duration_hours": round(est_hours, 2),
        "effective_config": effective_cfg,
    }

    if getattr(args, "json", False):
        reporter.print_json(plan_data)
        return int(ExitCode.SUCCESS)

    headers = ["Dimension", "Count / Value"]
    rows = [
        ["Modules Queued", str(mod_count)],
        ["Target Models", str(models_count)],
        ["Datasets", str(datasets_count)],
        ["Ranks", str(ranks_count)],
        ["Total Planned Runs", f"{total_runs:,}"],
        ["Estimated Duration", f"~{est_hours:.1f} hours (1x GPU)"],
    ]
    reporter.print_table("Benchmark Execution Plan", headers, rows)
    if total_runs > 500 and not getattr(args, "yes", False):
        reporter.log("⚠️  Grid size exceeds 500 runs. Use --yes to confirm execution.")
    return int(ExitCode.SUCCESS)


def handle_resume(args: argparse.Namespace) -> int:
    """Resume an interrupted benchmark run from an existing run directory."""
    out_dir = Path(getattr(args, "output_dir", "results/latest")).resolve()
    if not out_dir.exists():
        reporter = CliReporter()
        reporter.log(f"[-] Cannot resume: output directory does not exist: {out_dir}")
        return int(ExitCode.RUN_RESUME_MISMATCH)

    with run_directory_lock(out_dir):
        from neuralops.benchmarks.api import Runner
        runner = Runner(output_dir=str(out_dir), resume=True)
        runner.resume()
    return int(ExitCode.SUCCESS)


def handle_run(args: argparse.Namespace) -> int:
    """Execute benchmark run with deterministic config precedence and device validation."""
    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    if getattr(args, "recipe_cmd", None) == "recipe":
        from neuralops.cli.core.commands.results import handle_recipe
        args.recipe_action = "benchmark"
        args.file = getattr(args, "recipe_file", None)
        return handle_recipe(args)

    cli_dict = {k: v for k, v in vars(args).items() if v is not None}
    config_path = getattr(args, "config", None)
    cfg = load_effective_config(cli_dict, config_path=config_path)

    # Device resolution (Fail-closed on explicit CUDA)
    req_device = cfg.get("device", "auto")
    device_info = resolve_device(req_device)
    cfg["device"] = device_info.canonical_name

    out_dir = create_run_directory(
        output_dir=cfg.get("output_dir", "results/run"),
        run_id=cfg.get("run_id"),
        overwrite=bool(cfg.get("overwrite", False)),
    )

    with run_directory_lock(out_dir):
        if cfg.get("category") in ("cuda", "triton", "quantization", "moe", "serving", "rope", "sampling", "ssm", "varlen", "sweep"):
            from neuralops.benchmarks.runner import run_catalog_benchmarks
            reporter.log(f"[+] Launching {cfg['category']} benchmark runner on {device_info.canonical_name}...")
            results = run_catalog_benchmarks(
                category=cfg.get("category"),
                modules=cfg.get("modules") or cfg.get("module"),
                family=cfg.get("family"),
                warmup=int(cfg.get("warmup", 3)),
                iters=int(cfg.get("iters", 10)),
                seq_len=int(cfg.get("seq_len", 1024)),
                batch_size=int(cfg.get("batch_size", 2)),
                dim=int(cfg.get("dim", 768)),
                heads=int(cfg.get("heads", 12)),
                out_dir=str(out_dir),
                dynamic=bool(cfg.get("dynamic", False)),
                device=device_info.canonical_name,
                parallel_workers=int(cfg.get("parallel_workers", 4)),
                seed=int(cfg.get("seed", 42)),
                dtype=cfg.get("dtype", "fp32"),
                tolerance=cfg.get("tolerance"),
                limit=cfg.get("limit"),
            )
            if any(row.get("status") == "FAIL_CORRECTNESS" for row in results):
                raise ExecutionError("Correctness gate failed; timing was not accepted for every requested operation")
            return int(ExitCode.SUCCESS)

        # Standard attention/compressor run
        mods = "all"
        module_arg = cfg.get("module")
        modules_arg = cfg.get("modules")
        if cfg.get("no_compression"):
            mods = "baseline"
        elif module_arg:
            mods = [module_arg]
        elif modules_arg:
            mods = [m.strip() for m in modules_arg.split(",")] if isinstance(modules_arg, str) else modules_arg

        ranks_val = cfg.get("ranks")
        ranks = [int(r.strip()) for r in ranks_val.split(",")] if isinstance(ranks_val, str) else (ranks_val or cfg.get("rank"))
        models_val = cfg.get("models")
        models = [m.strip() for m in models_val.split(",")] if isinstance(models_val, str) else (models_val or cfg.get("model"))
        datasets_val = cfg.get("datasets")
        datasets = [d.strip() for d in datasets_val.split(",")] if isinstance(datasets_val, str) else (datasets_val or cfg.get("dataset"))

        from neuralops.benchmarks.api import Runner

        runner = Runner(
            modules=mods,
            delta=bool(cfg.get("all_delta", False)),
            gated=bool(cfg.get("all_gated", False)),
            svd=bool(cfg.get("all_svd", False)),
            ssm=bool(cfg.get("all_ssm", False)),
            linear=bool(cfg.get("all_linear", False)),
            sparse=bool(cfg.get("all_sparse", False)),
            hybrid=bool(cfg.get("all_hybrid", False)),
            all_attn=bool(cfg.get("all_attn", False)),
            all_comp=bool(cfg.get("all_comp", False)),
            all_modules=bool(cfg.get("all_modules", False)),
            fail_delta=bool(cfg.get("fail_delta", False)),
            errors_only=bool(cfg.get("errors_only", False)),
            models=models,
            datasets=datasets,
            ranks=ranks,
            parallel=bool(cfg.get("parallel", False)),
            output_dir=str(out_dir),
            epochs=int(cfg.get("epochs", 1)),
            batch_size=int(cfg.get("batch_size", 2)),
            lr=float(cfg.get("lr", 1e-4)),
            seq_len=int(cfg.get("seq_len", 1024)),
            device=device_info.canonical_name,
            timeout_s=int(cfg.get("timeout_s", 600)),
            eval_only=(cfg.get("training_type") == "none"),
            training_type=cfg.get("training_type", "none"),
            kd_teacher=cfg.get("kd_teacher"),
            kd_temperature=float(cfg.get("kd_temperature", 2.0)),
            kd_alpha=float(cfg.get("kd_alpha", 0.5)),
            recovery_steps=int(cfg.get("recovery_steps", 100)),
            task=cfg.get("task"),
            max_samples=int(cfg.get("max_samples", 1000)),
            max_eval_samples=int(cfg.get("max_eval_samples", 500)),
            grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
            tracker=(cfg.get("tracker") if cfg.get("tracker") != "none" else ("local" if cfg.get("track") else "none")),
            project_name=cfg.get("project_name", "neuralops"),
            run_name=cfg.get("run_name"),
            log_interval=int(cfg.get("log_interval", 10)),
            distributed_backend=cfg.get("distributed_backend", "none"),
            mixed_precision=cfg.get("mixed_precision", "none"),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            hf_token=cfg.get("hf_token"),
            language=cfg.get("language", "en"),
            text_column=cfg.get("text_column", "text"),
            label_column=cfg.get("label_column", "label"),
            id_column=cfg.get("id_column", "id"),
            preprocess_script=cfg.get("preprocess_script"),
            auto_commit=bool(cfg.get("auto_commit", False)) and not bool(cfg.get("no_auto_commit", False)),
            commit_interval=int(cfg.get("commit_interval", 5)),
            push_to_hf=bool(cfg.get("push_to_hf", False)),
            deduplicate=not bool(cfg.get("no_deduplicate", False)),
            force_rerun=bool(cfg.get("force_rerun", False)),
        )

        for res in runner.stream():
            status_symbol = "[+]" if res.status == "success" else "[-]"
            mod_name = res.module.get("name") if isinstance(res.module, dict) else res.module
            mdl_name = res.model.get("name") if isinstance(res.model, dict) else res.model
            ds_name = res.dataset.get("name") if isinstance(res.dataset, dict) else res.dataset
            reporter.log(f"{status_symbol} [{res.status.upper()}] {mod_name} on {mdl_name}/{ds_name}")

    return int(ExitCode.SUCCESS)
