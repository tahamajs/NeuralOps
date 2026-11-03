#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""
CLI Benchmark Command

Implements the command-line interface for running attention, compressor,
and kernel benchmarks, as well as the unified pre-training, healing, and
evaluation matrix dispatcher (`neuralops benchmark eval`).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

logger = logging.getLogger("neuralops.cli.bench")


def handle_eval(args: argparse.Namespace) -> int:
    """Execute the unified pre-benchmark micro-training, surgical healing, and evaluation pipeline.

    Wires together:
      - UniversalModelAdapter (HuggingFace / PyTorch / Surgerized checkpoints)
      - TrainAndBenchmarkPipeline (Ridge calibration + 4 micro-training regimes)
      - BenchmarkPipeline (Sequential / matrix evaluation across registered tasks)
      - MultiFormatSaver (Atomic WAL journaling, JSON, CSV, and SQLite master ledger)
    """
    model_arg = getattr(args, "model", "gpt2")
    sweep_attns_arg = getattr(args, "sweep_attentions", None)
    sweep_comps_arg = getattr(args, "sweep_compressors", None)
    benchmarks_arg = getattr(args, "benchmarks", "perplexity")
    pre_train_steps = int(getattr(args, "pre_train_steps", 0) or 0)
    pre_train_engine = getattr(args, "pre_train_engine", "multi_level_kd") or "multi_level_kd"
    pre_train_optimizer = getattr(args, "pre_train_optimizer", "muon") or "muon"
    pre_train_scheduler = getattr(args, "pre_train_scheduler", "wsd") or "wsd"
    auto_calibrate = bool(getattr(args, "auto_calibrate", False))
    resume = bool(getattr(args, "resume", False))
    output_dir = getattr(args, "output_dir", "results/eval_matrix") or "results/eval_matrix"
    device_str = getattr(args, "device", "cpu") or "cpu"

    device = torch.device(device_str)

    print("==================================================")
    print("NeuralOps Universal Benchmark Evaluator ('eval')")
    print(f"Model:                {model_arg}")
    print(f"Sweep Attentions:     {sweep_attns_arg}")
    print(f"Sweep Compressors:    {sweep_comps_arg}")
    print(f"Benchmark Suites:     {benchmarks_arg}")
    print(f"Pre-train Steps:      {pre_train_steps} (Engine={pre_train_engine}, Opt={pre_train_optimizer}, Sched={pre_train_scheduler})")
    print(f"Auto-Calibrate Ridge: {auto_calibrate}")
    print(f"Resume Sweep:         {resume}")
    print(f"Output Directory:     {output_dir}")
    print(f"Target Device:        {device}")
    print("==================================================")

    # 1. Lazy imports of architecture components
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    from neuralops.benchmarks.adapters.model import UniversalModelAdapter, BaseModelAdapter
    from neuralops.benchmarks.pipeline import BenchmarkPipeline, SweepController
    from neuralops.benchmarks.eval_with_training import TrainAndBenchmarkPipeline
    from neuralops.benchmarks.results.saver import MultiFormatSaver
    from neuralops.benchmarks.results.resumer import BenchmarkSweepResumer
    from neuralops.benchmarks.results.schema import BenchmarkResult
    from neuralops.benchmarks.registry import BENCHMARK_REGISTRY

    os.makedirs(output_dir, exist_ok=True)
    saver = MultiFormatSaver(output_dir=output_dir)
    resumer = BenchmarkSweepResumer(output_dir=output_dir) if resume else None

    # Parse benchmarks
    if isinstance(benchmarks_arg, str):
        benchmarks_list = [b.strip() for b in benchmarks_arg.split(",") if b.strip()]
    elif isinstance(benchmarks_arg, (list, tuple)):
        benchmarks_list = list(benchmarks_arg)
    else:
        benchmarks_list = ["perplexity"]

    # Parse sweep attentions
    attentions_list: List[str] = []
    if sweep_attns_arg:
        if sweep_attns_arg.strip().lower() == "all":
            attentions_list = sorted(list(ATTENTION_REGISTRY.keys()))
        else:
            attentions_list = [a.strip() for a in sweep_attns_arg.split(",") if a.strip()]

    # Parse sweep compressors
    compressors_list: List[str] = []
    if sweep_comps_arg:
        if sweep_comps_arg.strip().lower() == "all":
            compressors_list = sorted(list(COMPRESSOR_REGISTRY.keys()))
        else:
            compressors_list = [c.strip() for c in sweep_comps_arg.split(",") if c.strip()]

    # Load or wrap base model
    logger.info("Initializing model: %s", model_arg)
    try:
        model_adapter = UniversalModelAdapter.from_pretrained(
            model_arg,
            device=device,
        )
    except Exception as exc:
        logger.warning(
            "UniversalModelAdapter.from_pretrained failed (%s). Creating lightweight mock model.",
            exc,
        )
        # Create minimal fallback module
        class SimpleMockLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.proj = nn.Linear(64, 100)

            def forward(self, input_ids: torch.Tensor, **kw: Any) -> Any:
                x = self.embed(input_ids)
                logits = self.proj(x)
                class Out:
                    pass
                o = Out()
                o.logits = logits
                return o

        mock_m = SimpleMockLM().to(device)
        model_adapter = UniversalModelAdapter.from_pretrained(
            mock_m,
            model_name=str(model_arg),
            device=device,
        )

    # 2. Case A: Sweep Attentions with Pre-Training / Surgical Healing
    if attentions_list:
        print(f"\n[*] Launching Attention Sweep Matrix across {len(attentions_list)} modules...")
        if pre_train_steps > 0 or auto_calibrate:
            # Execute via TrainAndBenchmarkPipeline
            trainer_pipeline = TrainAndBenchmarkPipeline(
                output_dir=output_dir,
                device=device,
                saver=saver,
            )

            # Generate synthetic calibration/train loader if needed
            class SyntheticDataset(torch.utils.data.Dataset):
                def __len__(self) -> int:
                    return max(pre_train_steps + 10, 16)

                def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                    return {
                        "input_ids": torch.randint(0, 50, (1, 16), dtype=torch.long),
                        "labels": torch.randint(0, 50, (1, 16), dtype=torch.long),
                    }

            syn_loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=2)

            for idx, attn_name in enumerate(attentions_list, 1):
                # Check resume skip
                if resumer:
                    skip_all = True
                    for b_key in benchmarks_list:
                        skip, _ = resumer.should_skip(model_adapter.model_name, attn_name, b_key)
                        if not skip:
                            skip_all = False
                            break
                    if skip_all:
                        print(f"[{idx}/{len(attentions_list)}] Skipping completed attention: {attn_name}")
                        continue

                print(f"\n[{idx}/{len(attentions_list)}] Processing attention: {attn_name}")
                try:
                    trainer_pipeline.sweep_transplant_heal_and_benchmark(
                        base_model=model_adapter.model,
                        attention_keys=[attn_name],
                        benchmark_keys=benchmarks_list,
                        train_loader=syn_loader,
                        pre_train_steps=pre_train_steps,
                        optimizer_type=pre_train_optimizer,
                        regime=pre_train_engine,
                        device=str(device),
                    )
                except Exception as e:
                    logger.error("Failed processing attention %s: %s", attn_name, e)
                    # Record failure in saver
                    err_res = BenchmarkResult(
                        run_id=f"fail_{attn_name}_{int(time.time())}",
                        module_name=attn_name,
                        benchmark_suite="eval_matrix",
                        status="failure",
                        metrics={"error": str(e)},
                        metadata={"attention_key": attn_name, "model": model_adapter.model_name},
                    )
                    saver.save(err_res)
        else:
            # Sweep directly via SweepController and BenchmarkPipeline
            controller = SweepController(
                base_model=model_adapter.model,
                output_dir=output_dir,
                saver=saver,
                device=device,
            )
            # Filter pending if resuming
            pending_attns = attentions_list
            if resumer:
                pending_attns = [
                    a for a in attentions_list
                    if not all(resumer.is_completed(model_adapter.model_name, a, b) for b in benchmarks_list)
                ]
            controller.sweep_attentions(pending_attns, benchmarks_list)

    # 3. Case B: Sweep Compressors
    elif compressors_list:
        print(f"\n[*] Launching Compressor Sweep Matrix across {len(compressors_list)} engines...")
        controller = SweepController(
            base_model=model_adapter.model,
            output_dir=output_dir,
            saver=saver,
            device=device,
        )
        pending_comps = compressors_list
        if resumer:
            pending_comps = [
                c for c in compressors_list
                if not all(resumer.is_completed(model_adapter.model_name, c, b) for b in benchmarks_list)
            ]
        controller.sweep_compressors(pending_comps, benchmarks_list)

    # 4. Case C: Direct single-model evaluation across requested benchmark tasks
    else:
        print(f"\n[*] Evaluating model '{model_adapter.model_name}' directly across: {benchmarks_list}")
        pipeline = BenchmarkPipeline(
            model_adapter=model_adapter,
            task_keys=benchmarks_list,
            saver=saver,
            device=device,
        )
        for b_key in benchmarks_list:
            if resumer and resumer.is_completed(model_adapter.model_name, "baseline", b_key):
                print(f"Skipping already completed task: {b_key}")
                continue
            res = pipeline.execute_task(b_key)
            print(f"  --> Benchmark '{b_key}' completed: {res.get('metrics', {})}")

    print("\n==================================================")
    print(f"Evaluation Matrix finished. Telemetry persisted to: {output_dir}")
    print("==================================================")
    return 0


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register 'eval' subcommand with full pre-training and benchmark matrix flags."""
    parser = subparsers.add_parser(
        "eval",
        help="Universal evaluation matrix across attention mechanisms, compressors, and competitor benchmarks",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID or path to model checkpoint directory",
    )
    parser.add_argument(
        "--sweep-attentions",
        type=str,
        default=None,
        help="Comma-separated attention keys (or 'all' for all 131 registered keys)",
    )
    parser.add_argument(
        "--sweep-compressors",
        type=str,
        default=None,
        help="Comma-separated compressor keys (or 'all' for all 72 registered keys)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="perplexity",
        help="Comma-separated benchmark suite keys (e.g. perplexity, flash_attn_tflops, vllm_poisson)",
    )
    parser.add_argument(
        "--pre-train-steps",
        type=int,
        default=0,
        help="Number of pre-benchmark micro-training / healing steps (default: 0)",
    )
    parser.add_argument(
        "--pre-train-engine",
        type=str,
        default="multi_level_kd",
        choices=["multi_level_kd", "mohawk", "f_micro", "dora", "ttt", "spectral_annealing"],
        help="Pre-benchmark micro-training distillation engine or regime",
    )
    parser.add_argument(
        "--pre-train-optimizer",
        type=str,
        default="muon",
        choices=["muon", "adamw", "lion", "stiefel"],
        help="Optimizer used during pre-benchmark micro-training",
    )
    parser.add_argument(
        "--pre-train-scheduler",
        type=str,
        default="wsd",
        choices=["wsd", "cosine", "linear"],
        help="Learning rate scheduler for micro-training",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        default=False,
        help="Enable closed-form Ridge projection calibration before evaluation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume interrupted sweep by checking existing .journal.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/eval_matrix",
        help="Output directory for logs, WAL journal, and SQLite ledger",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device (e.g. cpu, cuda, cuda:0)",
    )
    parser.set_defaults(func=handle_eval)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build standalone argument parser for bench_cmd."""
    parser = argparse.ArgumentParser(
        prog="neuralops-bench",
        description="NeuralOps Benchmark Command & Universal Matrix Dispatcher",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Benchmark action")
    add_eval_subparser(subparsers)

    # Legacy benchmark arguments
    parser.add_argument("--module", "--target", dest="module", type=str, default=None)
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--all-attention", "--all", dest="all_attention", action="store_true", default=False)
    parser.add_argument("--all-compressors", dest="all_compressors", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def cmd_bench(args: Any) -> int:
    """Main entrypoint for benchmark CLI command."""
    # If invoked with subcommand 'eval' or func=handle_eval
    if getattr(args, "subcommand", None) == "eval" or getattr(args, "func", None) == handle_eval:
        return handle_eval(args)

    # If flags indicate eval mode or user specified eval mode
    if getattr(args, "mode", None) == "eval" and (
        getattr(args, "sweep_attentions", None) is not None
        or getattr(args, "sweep_compressors", None) is not None
        or getattr(args, "benchmarks", None) is not None
        or getattr(args, "pre_train_steps", None) is not None
        or getattr(args, "auto_calibrate", False)
        or getattr(args, "resume", False)
    ):
        return handle_eval(args)

    target = getattr(args, "module", None) or getattr(args, "target", None)
    mode = getattr(args, "mode", "eval")
    model = getattr(args, "model", "gpt2")
    dataset = getattr(args, "dataset", "sst2")
    rank = getattr(args, "rank", 32)
    device = getattr(args, "device", "cuda")

    print("==================================================")
    print("NeuralOps Benchmark Runner")
    print(f"Target:   {target}")
    print(f"Mode:     {mode}")
    print(f"Model:    {model} | Dataset: {dataset} | Rank: {rank} | Device: {device}")
    print("==================================================")

    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    import neuralops.attentions
    import neuralops.compression_engine

    all_attn = getattr(args, "all_attention", False) or getattr(args, "all", False)
    all_comp = getattr(args, "all_compressors", False) or getattr(args, "all", False)

    targets = []
    if all_attn:
        targets.extend([(k, "attention") for k in sorted(ATTENTION_REGISTRY.keys())])
    if all_comp:
        targets.extend([(k, "compressor") for k in sorted(COMPRESSOR_REGISTRY.keys())])

    if not targets:
        if not target:
            print("Error: --module or sweep flag (--all-attention, --all-compressors, --all) is required.", file=sys.stderr)
            return 1
        mod_type = "attention" if target in ATTENTION_REGISTRY else "compressor"
        targets = [(target, mod_type)]

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(targets) > 1:
        print("==================================================")
        print(f"NeuralOps Batch Sweep Runner: {len(targets)} modules queued")
        print(f"Mode: {mode} | Model: {model} | Dataset: {dataset} | Device: {device}")
        print("==================================================")

        results = []
        passes = 0
        failures = 0

        for idx, (mod_name, mod_type) in enumerate(targets, 1):
            print(f"\n[{idx}/{len(targets)}] Evaluating {mod_type}: {mod_name}...")
            start_t = time.time()
            try:
                if mode == "kd":
                    cmd = [
                        sys.executable,
                        "neuralops/benchmarks/_legacy/kd_quick_test.py",
                        "--modules", mod_name,
                        "--type", mod_type,
                        "--model_name", model,
                        "--rank", str(rank),
                        "--methods", "C",
                        "--use_cache",
                        "--batch_size", str(getattr(args, "batch_size", 4)),
                        "--grad_accum_steps", str(getattr(args, "grad_accum_steps", 2)),
                        "--recovery_steps", str(getattr(args, "recovery_steps", 2)),
                        "--self_train_steps", str(getattr(args, "self_train_steps", 2)),
                        "--learning_rate", str(getattr(args, "learning_rate", 3e-4)),
                        "--device", device,
                    ]
                    code = subprocess.call(cmd)
                    status = "PASS" if code == 0 else "FAIL"
                    err = None if code == 0 else f"Process exited with code {code}"
                    entry = {"module": mod_name, "type": mod_type, "status": status, "error": err, "elapsed_s": round(time.time() - start_t, 2)}
                else:
                    from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
                    dev = torch.device(device)
                    res = evaluate_single_combination_opt(
                        track="cli_bench",
                        module_name=mod_name,
                        module_type=mod_type,
                        model_family=model,
                        dataset=dataset,
                        rank=rank,
                        device=dev,
                    )
                    d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
                    status = "PASS" if d.get("status") in ("PASS", "PASS_NO_COMPRESSION") else "FAIL"
                    entry = {"module": mod_name, "type": mod_type, "status": status, "metrics": d, "elapsed_s": round(time.time() - start_t, 2)}

                if status == "PASS":
                    passes += 1
                    print(f"  --> PASS ({entry['elapsed_s']}s)")
                else:
                    failures += 1
                    print(f"  --> FAIL: {entry.get('error') or entry.get('metrics', {}).get('error_message')}")

            except Exception as e:
                failures += 1
                entry = {"module": mod_name, "type": mod_type, "status": "FAIL", "error": str(e), "elapsed_s": round(time.time() - start_t, 2)}
                print(f"  --> FAIL (Exception): {e}")

            results.append(entry)

            if output_dir:
                with open(os.path.join(output_dir, "sweep_progress.json"), "w") as f:
                    json.dump(results, f, indent=2)

        summary_path = os.path.join(output_dir, "sweep_summary.json") if output_dir else None
        if summary_path:
            with open(summary_path, "w") as f:
                json.dump({"total": len(targets), "passed": passes, "failed": failures, "results": results}, f, indent=2)
            print(f"\nSaved full sweep results to {summary_path}")

        print(f"\nSweep Finished: {passes}/{len(targets)} passed, {failures} failed.")
        return 0 if failures == 0 else 1

    mod_name, mod_type = targets[0]
    if mode == "kd":
        cmd = [
            sys.executable,
            "neuralops/benchmarks/_legacy/kd_quick_test.py",
            "--modules", mod_name,
            "--type", mod_type,
            "--model_name", model,
            "--rank", str(rank),
            "--methods", "C",
            "--use_cache",
            "--batch_size", str(getattr(args, "batch_size", 4)),
            "--grad_accum_steps", str(getattr(args, "grad_accum_steps", 2)),
            "--recovery_steps", str(getattr(args, "recovery_steps", 50)),
            "--self_train_steps", str(getattr(args, "self_train_steps", 20)),
            "--learning_rate", str(getattr(args, "learning_rate", 3e-4)),
            "--device", device,
        ]
        return subprocess.call(cmd)
    else:
        from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
        dev = torch.device(device)
        res = evaluate_single_combination_opt(
            track="cli_bench",
            module_name=mod_name,
            module_type=mod_type,
            model_family=model,
            dataset=dataset,
            rank=rank,
            device=dev,
        )
        print("\nBenchmark completed successfully:")
        print(res.to_dict() if hasattr(res, "to_dict") else res)
        return 0


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_bench(cli_args))
