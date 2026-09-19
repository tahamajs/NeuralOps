#!/usr/bin/env python3
"""
Master Orchestrator for All New ICLR 2027 Benchmark Blocks.

Sequentially executes or dry-runs:
  1. Baseline Comparison (--baselines)
  2. Incremental Ablation Study (--ablation)
  3. Multi-Seed Statistical Rigor (--multiseed)
  4. Sequence-Length Scaling (--seqlen)
  5. Complete Suite (--all)

Usage:
    python run_all_new_benchmarks.py --dry_run --all
    python run_all_new_benchmarks.py --baselines
    python run_all_new_benchmarks.py --ablation
    python run_all_new_benchmarks.py --multiseed
    python run_all_new_benchmarks.py --seqlen
    python run_all_new_benchmarks.py --all
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent

LOG = logging.getLogger("MasterBenchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master runner for newly prepared ICLR benchmark blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baselines", action="store_true", default=False,
                        help="Run Block 1: Literature Baseline Comparison")
    parser.add_argument("--ablation", action="store_true", default=False,
                        help="Run Block 2: 5-Stage Incremental Pipeline Ablation")
    parser.add_argument("--multiseed", action="store_true", default=False,
                        help="Run Block 3A: Multi-Seed Statistical Rigor Trials")
    parser.add_argument("--seqlen", action="store_true", default=False,
                        help="Run Block 3B: Sequence-Length Scaling Analysis")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Run all four evaluation blocks sequentially")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print sequence of commands without executing")
    parser.add_argument("--rank", type=int, default=64,
                        help="Decomposition rank parameter")
    return parser.parse_args()


def run_stage(cmd: list[str], dry_run: bool = False, stage_name: str = "") -> int:
    cmd_str = " ".join(cmd)
    print("\n" + "=" * 80)
    print(f"[STAGE: {stage_name}]")
    print(f"Command: {cmd_str}")
    print("=" * 80)

    if dry_run:
        print(f"[DRY RUN] Would execute: {cmd_str}")
        return 0

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"[STAGE COMPLETE] {stage_name} succeeded in {elapsed:.1f}s")
        return proc.returncode
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"[STAGE FAILED] {stage_name} failed with code {e.returncode} after {elapsed:.1f}s")
        return e.returncode


def main():
    args = parse_args()
    py_exec = sys.executable

    run_baselines = args.baselines or args.all
    run_ablation = args.ablation or args.all
    run_multiseed = args.multiseed or args.all
    run_seqlen = args.seqlen or args.all

    if not any([run_baselines, run_ablation, run_multiseed, run_seqlen]):
        print("No evaluation block selected. Specify at least one flag (--baselines, --ablation, --multiseed, --seqlen, or --all).")
        return

    print("=" * 80)
    print("MASTER ICLR BENCHMARK ORCHESTRATOR")
    print(f"Baselines:   {run_baselines}")
    print(f"Ablation:    {run_ablation}")
    print(f"Multi-Seed:  {run_multiseed}")
    print(f"SeqLen:      {run_seqlen}")
    print(f"Rank:        {args.rank}")
    print(f"Dry Run:     {args.dry_run}")
    print("=" * 80)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 1. Literature Baseline Comparison
    if run_baselines:
        cmd = [py_exec, "Codes/framework/run_baseline_comparison.py", "--rank", str(args.rank)]
        if args.dry_run:
            cmd.append("--dry_run")
        run_stage(cmd, dry_run=args.dry_run, stage_name="Baseline Comparison (Block 1)")

    # 2. 5-Stage Incremental Pipeline Ablation
    if run_ablation:
        cmd = [py_exec, "Codes/framework/run_ablation.py", "--rank", str(args.rank)]
        if args.dry_run:
            cmd.append("--dry_run")
        run_stage(cmd, dry_run=args.dry_run, stage_name="Incremental Ablation Study (Block 2)")

    # 3. Multi-Seed Statistical Rigor
    if run_multiseed:
        cmd = [py_exec, "Codes/framework/run_multiseed.py", "--rank", str(args.rank)]
        if args.dry_run:
            cmd.append("--dry_run")
        run_stage(cmd, dry_run=args.dry_run, stage_name="Multi-Seed Evaluation (Block 3A)")

    # 4. Sequence-Length Scaling Analysis
    if run_seqlen:
        cmd = [py_exec, "Codes/framework/run_seqlen_scaling.py", "--rank", str(args.rank)]
        if args.dry_run:
            cmd.append("--dry_run")
        run_stage(cmd, dry_run=args.dry_run, stage_name="Sequence-Length Scaling (Block 3B)")

        # Run scaling plotter after execution
        plot_cmd = [py_exec, "Codes/framework/plot_seqlen_scaling.py"]
        run_stage(plot_cmd, dry_run=args.dry_run, stage_name="Sequence-Length Scaling Plotter")

    print("\n" + "=" * 80)
    print("ALL REQUESTED BENCHMARK BLOCKS PROCESSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
