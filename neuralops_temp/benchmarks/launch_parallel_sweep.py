#!/usr/bin/env python3
"""
Launch and supervise 4 parallel GPT-2 sweep workers.

Usage:
    python neuralops/launch_parallel_sweep.py --total_workers 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Supervise parallel sweep workers.")
    parser.add_argument("--total_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/06_adhoc_experiments/perplexity_seqlen/gpt2_wikitext2_sweep",
        help="Path to output directory.",
    )
    args, unknown = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Launcher] Launching {args.total_workers} parallel workers...")
    print(f"[Launcher] Project root: {project_root}")
    print(f"[Launcher] Output directory: {args.output_dir}")

    procs = []
    log_files = []

    for worker_id in range(args.total_workers):
        out_log_path = logs_dir / f"worker_{worker_id}.out"
        err_log_path = logs_dir / f"worker_{worker_id}.err"
        worker_log_file = logs_dir / f"worker_{worker_id}.log"

        out_f = open(out_log_path, "w", encoding="utf-8")
        err_f = open(err_log_path, "w", encoding="utf-8")
        log_files.extend([out_f, err_f])

        cmd = [
            sys.executable,
            "-u",
            str(project_root / "neuralops" / "benchmarks" / "run_gpt2_sweep.py"),
            "--output_dir",
            args.output_dir,
            "--total_workers",
            str(args.total_workers),
            "--worker_id",
            str(worker_id),
            "--log_file",
            str(worker_log_file),
        ] + unknown

        print(f"[Launcher] Spawning Worker {worker_id}: {' '.join(cmd)}")
        p = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            cwd=str(project_root),
        )
        procs.append((worker_id, p))

    print(f"[Launcher] All {args.total_workers} workers launched.")

    try:
        while True:
            alive = [(wid, p) for (wid, p) in procs if p.poll() is None]
            if not alive:
                print("[Launcher] All workers have completed.")
                break
            alive_ids = [wid for wid, _ in alive]
            print(f"[Launcher] Active workers: {alive_ids} ({len(alive)}/{args.total_workers})")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted! Terminating child workers...")
        for wid, p in procs:
            if p.poll() is None:
                print(f"[Launcher] Killing Worker {wid} (PID {p.pid})...")
                p.terminate()
        for wid, p in procs:
            p.wait()
    finally:
        for f in log_files:
            try:
                f.close()
            except Exception:
                pass

    # Print summary status
    print("\n[Launcher] Worker exit codes:")
    for wid, p in procs:
        print(f"  Worker {wid} (PID {p.pid}): exit code {p.returncode}")


if __name__ == "__main__":
    main()
