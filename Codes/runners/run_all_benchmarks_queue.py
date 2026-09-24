#!/usr/bin/env python3
"""
Master Benchmark Orchestrator for SVD Linear Attention & Compression Engines.

Runs each module benchmark (`run_all_4_<module>.py`) one-by-one sequentially:
  - Enforces uniform settings: --recovery_steps 50, pure inference profiling
  - Runs exactly one module at a time to protect GPU memory and ensure accurate latency metrics
  - Robust error handling: captures exit codes, logs errors, isolates failures without killing queue
  - Skip functionality: can resume from interruptions and skip modules that are already evaluated
  - Generates unified tracking summary across all executed modules (CSV & JSON)
  - Real-time console progress and execution heartbeat
"""

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


DEPRECATED_MODULES = {"tt", "tt_linear"}


def get_all_runner_scripts(repo_dir: Path) -> List[Path]:
    """Find all run_all_4_*.py files in the repository, excluding deprecated modules."""
    search_dirs = [repo_dir]
    for sub in [
        repo_dir / "Codes" / "runners",
        repo_dir / "runners",
        repo_dir / "Codes" / "experimental" / "runners",
        Path(__file__).resolve().parent,
    ]:
        if sub.is_dir() and sub not in search_dirs:
            search_dirs.append(sub)

    runners_map: Dict[str, Path] = {}
    for d in search_dirs:
        for p in sorted(d.glob("run_all_4_*.py")):
            name = extract_module_name(p)
            if name not in DEPRECATED_MODULES and name not in runners_map:
                runners_map[name] = p

    return [runners_map[k] for k in sorted(runners_map.keys())]


def extract_module_name(runner_path: Path) -> str:
    """Extract module identifier from runner filename: run_all_4_<name>.py -> <name>."""
    stem = runner_path.stem
    if stem.startswith("run_all_4_"):
        return stem[len("run_all_4_"):]
    return stem


def is_module_already_done(module_name: str, results_root: Path, required_models: List[str]) -> bool:
    """Check if all required models have already completed with summary.csv for this module and passed."""
    module_dir = results_root / f"unified_{module_name}"
    if not module_dir.is_dir():
        return False

    for m in required_models:
        csv_file = module_dir / m / "summary.csv"
        if not csv_file.is_file() or csv_file.stat().st_size == 0:
            return False
            
        # Check if the status was PASS_LOW or FAIL, if so we want to re-run
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return False
                # If any run for this model wasn't a PASS, we re-run
                if any(r.get('status') in ('FAIL', 'PASS_LOW') for r in rows):
                    return False
        except Exception:
            return False

    return True


def run_single_module(
    python_exec: str,
    runner_path: Path,
    models: List[str],
    recovery_steps: int,
    batch_size: Optional[int],
    timeout_s: int = 7200,
    log_dir: Optional[Path] = None,
    echo_stdout: bool = True,
) -> Dict[str, Any]:
    """Execute a single run_all_4_<module>.py runner with standard arguments."""
    module_name = extract_module_name(runner_path)
    cmd = [
        python_exec,
        "-u",
        str(runner_path.resolve()),
        "--models",
        ",".join(models),
        "--recovery_steps",
        str(recovery_steps),
    ]
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])

    start_time = time.time()
    log_file_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"benchmark_{module_name}.log"

    if echo_stdout:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> STARTING BENCHMARK: {module_name}", flush=True)
        print(f"Command: {' '.join(cmd)}", flush=True)

    process = None
    exit_code = -1
    error_msg = None

    try:
        if log_file_path:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with open(log_file_path, "w", encoding="utf-8", errors="replace") as f:
                for line in process.stdout:
                    if echo_stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    f.write(line)
                    f.flush()
            process.wait(timeout=timeout_s)
            exit_code = process.returncode
        else:
            proc = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
            exit_code = proc.returncode

    except subprocess.TimeoutExpired:
        error_msg = f"Timed out after {timeout_s} seconds."
        print(f"\n[ERROR] Module {module_name} timed out ({timeout_s}s). Terminating...", flush=True)
        if process:
            process.kill()
        exit_code = -999
    except Exception as exc:
        error_msg = str(exc)
        print(f"\n[ERROR] Exception while running {module_name}: {exc}", flush=True)
        exit_code = -1

    # Prune any heavy .pt tensor files produced during the run to preserve disk space
    try:
        results_dir = runner_path.parent.parent.parent / "results"
        for candidate_dir in [results_dir / f"unified_{module_name}", results_dir / module_name]:
            if candidate_dir.exists():
                for pt_file in candidate_dir.glob("**/*.pt"):
                    try:
                        pt_file.unlink()
                    except Exception:
                        pass
    except Exception:
        pass

    elapsed_s = round(time.time() - start_time, 2)
    status_str = "SUCCESS" if exit_code == 0 else f"FAILED (exit: {exit_code})"
    if echo_stdout:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< FINISHED BENCHMARK: {module_name} -> {status_str} in {elapsed_s:.1f}s\n", flush=True)

    return {
        "module": module_name,
        "runner": runner_path.name,
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "status": "PASS" if exit_code == 0 else "FAIL",
        "error": error_msg,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Master Benchmark Orchestrator: run all module benchmarks one by one"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="distilbert,roberta,gpt2",
        help="Comma-separated list of models to evaluate (default: distilbert,roberta,gpt2)",
    )
    parser.add_argument(
        "--recovery_steps",
        type=int,
        default=50,
        help="Number of recovery fine-tuning steps (strictly enforced default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override for all runs (default: None, runner default applies)",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        help="Module name or prefix to start running from",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated specific list of modules to run (e.g. aasvd,eat,lora_attention)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip modules whose results already exist in results/unified_<module> (default: True)",
    )
    parser.add_argument(
        "--no_skip",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing module runs; re-run everything",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds per module benchmark (default: 3600s / 1 hour)",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable to invoke runners with (default: current interpreter)",
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=None,
        help="Path to project repository root (default: current working directory)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel module benchmarks to run simultaneously (default: 3)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of module names to exclude from execution",
    )

    args = parser.parse_args()

    repo_dir = Path(args.repo_dir) if args.repo_dir else Path.cwd()
    results_root = repo_dir / "results"
    log_dir = repo_dir / "benchmark_logs"
    models_list = [m.strip() for m in args.models.split(",") if m.strip()]

    all_runners = get_all_runner_scripts(repo_dir)
    if not all_runners:
        print(f"No run_all_4_*.py scripts found in {repo_dir}")
        sys.exit(1)

    exclude_set = {m.strip() for m in args.exclude.split(",") if m.strip()} if args.exclude else set()

    # Filter by specific modules if requested
    if args.modules:
        target_set = {m.strip() for m in args.modules.split(",") if m.strip()}
        runners = [r for r in all_runners if extract_module_name(r) in target_set and extract_module_name(r) not in exclude_set]
    else:
        runners = [r for r in all_runners if extract_module_name(r) not in exclude_set]

    # Filter starting point if requested
    if args.start_from:
        start_idx = 0
        for i, r in enumerate(runners):
            if extract_module_name(r) >= args.start_from:
                start_idx = i
                break
        runners = runners[start_idx:]

    print("=" * 80)
    print("MASTER BENCHMARK QUEUE ORCHESTRATOR")
    print("=" * 80)
    print(f"Repository Root : {repo_dir}")
    print(f"Total Runners   : {len(runners)} (out of {len(all_runners)} total)")
    print(f"Parallel Workers: {args.parallel}")
    print(f"Target Models   : {', '.join(models_list)}")
    print(f"Recovery Steps  : {args.recovery_steps}")
    print(f"Batch Size      : {args.batch_size or 'Module Default'}")
    print(f"Skip Existing   : {args.skip_existing}")
    print(f"Timeout / Module: {args.timeout}s")
    print(f"Log Directory   : {log_dir}")
    print("=" * 80, flush=True)

    queue_results: List[Dict[str, Any]] = []
    summary_csv_path = results_root / "master_queue_benchmark_summary.csv"
    summary_json_path = results_root / "master_queue_benchmark_summary.json"

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    total_modules = len(runners)
    overall_start_time = time.time()

    write_lock = threading.Lock()
    active_count = 0
    active_lock = threading.Lock()

    def process_module(item):
        nonlocal passed_count, failed_count, skipped_count, active_count
        idx, runner = item
        module_name = extract_module_name(runner)

        if args.skip_existing and is_module_already_done(module_name, results_root, models_list):
            with write_lock:
                skipped_count += 1
                res = {
                    "module": module_name,
                    "runner": runner.name,
                    "exit_code": 0,
                    "elapsed_s": 0.0,
                    "status": "SKIPPED_EXISTING",
                    "error": None,
                    "timestamp": datetime.now().isoformat(),
                }
                queue_results.append(res)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{idx}/{total_modules}] SKIPPED: '{module_name}' (already done & passed)", flush=True)
            return

        with active_lock:
            active_count += 1
            curr_active = active_count

        echo = (args.parallel == 1)
        with write_lock:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{idx}/{total_modules}] >>> START '{module_name}' (active: {curr_active}/{args.parallel})", flush=True)

        result = run_single_module(
            python_exec=args.python_exec,
            runner_path=runner,
            models=models_list,
            recovery_steps=args.recovery_steps,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
            log_dir=log_dir,
            echo_stdout=echo,
        )

        with active_lock:
            active_count -= 1

        with write_lock:
            queue_results.append(result)
            if result["status"] == "PASS":
                passed_count += 1
            else:
                failed_count += 1

            total_done = passed_count + failed_count + skipped_count
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [{total_done}/{total_modules}] <<< FINISH '{module_name}' -> {result['status']} in {result['elapsed_s']:.1f}s (Passed: {passed_count}, Failed: {failed_count}, Skipped: {skipped_count})",
                flush=True,
            )

            results_root.mkdir(parents=True, exist_ok=True)
            try:
                import pandas as pd
                pd.DataFrame(queue_results).to_csv(summary_csv_path, index=False)
            except Exception:
                pass
            try:
                with open(summary_json_path, "w", encoding="utf-8") as f:
                    json.dump(queue_results, f, indent=2)
            except Exception:
                pass

    indexed_runners = list(enumerate(runners, 1))

    if args.parallel > 1:
        print(f"\nLaunching parallel queue execution with {args.parallel} concurrent workers...\n", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            list(executor.map(process_module, indexed_runners))
    else:
        for item in indexed_runners:
            process_module(item)

    total_wall_s = round(time.time() - overall_start_time, 2)
    print("\n" + "=" * 80)
    print("ALL MODULE BENCHMARKS IN QUEUE COMPLETE!")
    print(f"Total Processed : {total_modules}")
    print(f"Passed          : {passed_count}")
    print(f"Failed          : {failed_count}")
    print(f"Skipped         : {skipped_count}")
    print(f"Total Time      : {total_wall_s:.1f}s ({total_wall_s/3600:.2f} hours)")
    print(f"Summary CSV     : {summary_csv_path}")
    print(f"Summary JSON    : {summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
