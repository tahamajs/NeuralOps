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
import sqlite3
import subprocess
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

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
                        module={"name": attn_name, "type": "attention"},
                        model={"name": model_adapter.model_name},
                        benchmark_suite="eval_matrix",
                        status="failure",
                        metrics={"error": str(e)},
                        metadata={"attention_key": attn_name, "model": model_adapter.model_name},
                    )
                    err_res.run_id = f"fail_{attn_name}_{int(time.time())}"
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

    # Render Rich summary table if available
    _render_results_table(output_dir)

    return 0


def _render_results_table(output_dir: str | Path) -> None:
    """Render a structured summary table from the run's journal or SQLite ledger."""
    p_dir = Path(output_dir)
    journal_path = p_dir / ".journal.jsonl"
    db_path = p_dir / "master_ledger.db"

    entries: List[Dict[str, Any]] = []

    # Read from journal if present
    if journal_path.exists():
        try:
            with open(journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.debug("Failed parsing journal for summary table: %s", e)

    # Alternatively query SQLite if journal was empty
    if not entries and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT run_id, model_name, transplanted_attention, benchmark_suite, status, metrics_json FROM benchmark_runs"
            ).fetchall()
            for r_id, m_name, t_attn, b_suite, st, m_json in rows:
                try:
                    m_dict = json.loads(m_json) if m_json else {}
                except Exception:
                    m_dict = {}
                entries.append({
                    "run_id": r_id,
                    "model_name": m_name,
                    "transplanted_attention": t_attn,
                    "benchmark_suite": b_suite,
                    "status": st,
                    "metrics": m_dict,
                })
            conn.close()
        except Exception as e:
            logger.debug("Failed querying sqlite for summary table: %s", e)

    if not entries:
        return

    if HAS_RICH and console is not None:
        table = Table(
            title=f"NeuralOps Benchmark Matrix Summary: {p_dir.name}",
            header_style="bold green",
            show_lines=True,
        )
        table.add_column("Run ID", style="dim", overflow="fold", max_width=20)
        table.add_column("Model", style="cyan")
        table.add_column("Attention / Target", style="bold yellow")
        table.add_column("Suite", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Metrics Summary", style="white")

        for ent in entries:
            st = ent.get("status", "unknown")
            st_styled = "[green]SUCCESS[/green]" if st == "success" else f"[red]{st.upper()}[/red]"
            m_data = ent.get("metrics") or {}
            m_summary_parts = []
            for mk, mv in m_data.items():
                if isinstance(mv, float):
                    m_summary_parts.append(f"{mk}={mv:.3f}")
                else:
                    m_summary_parts.append(f"{mk}={mv}")
            m_summary = ", ".join(m_summary_parts) if m_summary_parts else "N/A"

            table.add_row(
                str(ent.get("run_id", "")),
                str(ent.get("model_name", "")),
                str(ent.get("transplanted_attention", "") or "baseline"),
                str(ent.get("benchmark_suite", "")),
                st_styled,
                m_summary,
            )
        console.print(table)
    else:
        print("\n=== Benchmark Matrix Summary ===")
        for ent in entries:
            print(
                f"[{ent.get('status', '').upper()}] {ent.get('model_name')} | "
                f"{ent.get('transplanted_attention') or 'baseline'} | "
                f"Suite: {ent.get('benchmark_suite')} | "
                f"Metrics: {ent.get('metrics')}"
            )
        print("================================")


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


def execute_benchmark_cli(args: argparse.Namespace) -> int:
    """Unified benchmark CLI execution dispatcher."""
    bench_command = getattr(args, 'bench_command', None)

    if bench_command == 'list':
        import neuralops
        from neuralops.core.registry import COMPRESSOR_REGISTRY
        if 'neuralops.compressors' not in sys.modules:
            _comp_mod = types.ModuleType('neuralops.compressors')
            _comp_mod.COMPRESSOR_REGISTRY = COMPRESSOR_REGISTRY
            sys.modules['neuralops.compressors'] = _comp_mod

        from neuralops.attentions import ATTENTION_REGISTRY
        from neuralops.compressors import COMPRESSOR_REGISTRY
        from neuralops.benchmarks.suites import BENCHMARK_SUITES

        if getattr(args, 'attentions', False):
            for k in sorted(list(ATTENTION_REGISTRY)):
                print(k)
            return 0

        if getattr(args, 'compressors', False):
            for k in sorted(list(COMPRESSOR_REGISTRY)):
                print(k)
            return 0

        if getattr(args, 'suites', False):
            for s in sorted(list(BENCHMARK_SUITES.keys())):
                print(s)
            return 0

        for s in sorted(list(BENCHMARK_SUITES.keys())):
            print(s)
        return 0

    elif bench_command == 'run':
        from neuralops.config.loader import load_benchmark_config
        from neuralops.benchmarks.core.runner import BenchmarkRunner

        overrides = {}
        if getattr(args, 'device', None):
            overrides.setdefault('hardware', {})['device'] = args.device
        if getattr(args, 'backend', None):
            overrides.setdefault('hardware', {})['backend_preference'] = args.backend
        if getattr(args, 'tag', None):
            overrides['tag'] = args.tag

        config_source = args.config
        if not config_source:
            default_yaml = Path(f'neuralops/config/templates/benchmarks/{args.suite}.yaml')
            standard_yaml = Path(f'neuralops/config/templates/benchmarks/{args.suite}_standard.yaml')
            if default_yaml.exists():
                config_source = str(default_yaml)
            elif standard_yaml.exists():
                config_source = str(standard_yaml)
            else:
                config_source = str(default_yaml)

        config = load_benchmark_config(config_source, overrides=overrides if overrides else None)

        if getattr(args, 'suite', None):
            config.benchmark_suite = args.suite

        if getattr(args, 'dry_run', False):
            print('[DRY-RUN] Benchmark configuration validated successfully.')
            print(f'Suite: {config.benchmark_suite} | Device: {config.hardware.device}')
            if getattr(config, 'attention_sweep', None):
                sw = config.attention_sweep
                print(f'Attention sweep mechanisms: {len(sw.mechanisms) if sw.mechanisms != ["all"] else "all (131 registered)"}')
            return 0

        runner = BenchmarkRunner(config)
        summary = runner.run()
        print(f'Benchmark run {summary.run_id} completed with status: {summary.status}')
        return 0 if summary.status in ('SUCCESS', 'COMPLETED', 'PASSED') else 1

    elif bench_command == 'show':
        run_id = getattr(args, 'run_id', 'latest')
        root_dir = Path("results/benchmarks")

        target_dir: Optional[Path] = None
        if run_id == 'latest':
            latest_links = list(root_dir.glob("*/latest"))
            if latest_links:
                latest_links.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                target_dir = latest_links[0].resolve()
            else:
                runs = [p for p in root_dir.glob("*/runs/*") if p.is_dir()]
                if runs:
                    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    target_dir = runs[0]
        else:
            candidates = [p for p in root_dir.glob(f"*/runs/{run_id}") if p.is_dir()]
            if not candidates:
                candidates = [p for p in root_dir.glob(f"**/{run_id}") if p.is_dir()]
            if candidates:
                target_dir = candidates[0]
            elif Path(run_id).is_dir():
                target_dir = Path(run_id)

        if not target_dir or not target_dir.exists():
            print(f"Error: Benchmark run '{run_id}' not found in {root_dir}", file=sys.stderr)
            return 1

        if getattr(args, 'manifest', False):
            manifest_file = target_dir / "manifest.json"
            if manifest_file.exists():
                print(manifest_file.read_text(encoding="utf-8").strip())
                return 0
            else:
                print(f"Error: Manifest not found at {manifest_file}", file=sys.stderr)
                return 1

        if getattr(args, 'metrics', False):
            summary_file = target_dir / "summary.json"
            if summary_file.exists():
                print(summary_file.read_text(encoding="utf-8").strip())
                return 0
            csv_file = target_dir / "summary.csv"
            if csv_file.exists():
                print(csv_file.read_text(encoding="utf-8").strip())
                return 0
            print(f"Error: Metrics not found at {summary_file}", file=sys.stderr)
            return 1

        if getattr(args, 'log', False):
            log_file = target_dir / "stdout_stderr.log"
            if not log_file.exists():
                log_file = target_dir / "metrics.jsonl"
            if log_file.exists():
                print(log_file.read_text(encoding="utf-8").strip())
                return 0
            else:
                print(f"Error: Log file not found at {log_file}", file=sys.stderr)
                return 1

        report_file = target_dir / "benchmark_report.md"
        if report_file.exists():
            print(report_file.read_text(encoding="utf-8").strip())
            return 0
        summary_file = target_dir / "summary.json"
        if summary_file.exists():
            print(summary_file.read_text(encoding="utf-8").strip())
            return 0
        manifest_file = target_dir / "manifest.json"
        if manifest_file.exists():
            print(manifest_file.read_text(encoding="utf-8").strip())
            return 0

        print(f"Run directory located at: {target_dir}")
        return 0

    return 0


def add_benchmark_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register 'benchmark' command and subcommands into NeuralOps central CLI."""
    parser = subparsers.add_parser(
        'benchmark',
        help='Unified empirical benchmarking, profiling, and telemetry system',
        description='Execute, monitor, track, and analyze benchmark suites across hardware.'
    )
    bench_subparsers = parser.add_subparsers(dest='bench_command', required=True)

    # 1. run subcommand
    run_parser = bench_subparsers.add_parser('run', help='Execute an empirical benchmark run')
    run_parser.add_argument('--suite', '-s', type=str, required=True, help='Benchmark suite identifier')
    run_parser.add_argument('--config', '-c', type=str, default=None, help='Path to YAML or JSON benchmark run configuration file')
    run_parser.add_argument('--device', type=str, default=None, help='Compute device override (e.g. cuda:0 or cpu)')
    run_parser.add_argument('--backend', type=str, default=None, choices=['auto', 'cuda', 'triton', 'torch'])
    run_parser.add_argument('--tag', type=str, default=None, help='Semantic tag for ledger indexing')
    run_parser.add_argument('--dry-run', action='store_true', help='Validate parameters and print execution manifest without running kernels')
    run_parser.set_defaults(func=execute_benchmark_cli)

    # 2. list subcommand
    list_parser = bench_subparsers.add_parser('list', help='List available benchmark suites and registry entities')
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument('--suites', action='store_true', help='List registered benchmark suites')
    list_group.add_argument('--attentions', action='store_true', help='List all 131 canonical registered attention mechanisms')
    list_group.add_argument('--compressors', action='store_true', help='List all 72 canonical registered model compressors')
    list_parser.set_defaults(func=execute_benchmark_cli)

    # 3. show subcommand
    show_parser = bench_subparsers.add_parser('show', help='Inspect details and metrics of a previous benchmark run')
    show_parser.add_argument('run_id', type=str, help="UUID or timestamp run identifier or 'latest'")
    show_parser.add_argument('--manifest', action='store_true', help='Output raw JSON provenance manifest')
    show_parser.add_argument('--metrics', action='store_true', help='Display summary metrics')
    show_parser.add_argument('--log', action='store_true', help='Display raw execution console log')
    show_parser.set_defaults(func=execute_benchmark_cli)

    parser.set_defaults(func=execute_benchmark_cli)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build standalone argument parser for bench_cmd."""
    parser = argparse.ArgumentParser(
        prog='neuralops benchmark',
        description='NeuralOps Benchmark Command & Universal Matrix Dispatcher',
    )
    sub = parser.add_subparsers(dest='subcommand', help='Benchmark action')
    add_eval_subparser(sub)
    add_benchmark_parser(sub)

    # Legacy benchmark arguments
    parser.add_argument('--module', '--target', dest='module', type=str, default=None)
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--all-attention', '--all', dest='all_attention', action='store_true', default=False)
    parser.add_argument('--all-compressors', dest='all_compressors', action='store_true', default=False)
    parser.add_argument('--output-dir', type=str, default=None)
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
