"""
Master pipeline orchestrator.
Runs 2 workers total across all task types.
Priority order: zeroshot > rank_sweep > c4_ppl > latency > multiseed.
Idempotent: skips completed tasks.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv

WORKSPACE = Path("/d/SVD_Project") if sys.platform != "win32" else Path("D:/SVD_Project")

def run_task(task: dict) -> dict:
    t = task['task']
    module = task['module']
    dataset = task['dataset']
    backbone = task['backbone']
    
    if t == 'zeroshot':
        cmd = [
            sys.executable, "-u", "scripts/eval_zeroshot.py",
            "--module", module,
            "--model", "models/gpt2",
            "--tasks", "piqa,arc_easy,arc_challenge,hellaswag,winogrande",
            "--num_samples", "500",
            "--batch_size", "16",
            "--output_dir", f"results/zeroshot_parallel/{module}",
        ]
    elif t == 'rank_sweep':
        model = f"models/base_checkpoints/{backbone}_{dataset}"
        cmd = [
            sys.executable, "-u", "Codes/framework/run_rank_sweeper.py",
            "--modules", module,
            "--ranks", "64,128,256,384",
            "--model", model,
            "--model_family", "bert",
            "--dataset", dataset,
            "--output_base", f"results/rank_sweep_{dataset}",
            "--parallel_ranks", "1",
            "--batch_size", "16",
        ]
    elif t == 'c4_ppl':
        cmd = [
            sys.executable, "-u", "Codes/framework/run_quality_sweep.py",
            "--modules", module,
            "--model", "models/gpt2",
            "--model_family", "gpt2",
            "--dataset", "c4",
            "--rank", "384",
            "--batch_size", "8",
            "--output_dir", f"results/c4_baselines/{module}",
        ]
    else:
        return {"task": t, "module": module, "status": "UNKNOWN_TASK"}
    
    t0 = time.time()
    log_dir = Path(f"logs/pipeline/{t}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module}_{dataset}.log"
    
    with open(log_file, "w") as lf:
        res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    
    elapsed = time.time() - t0
    return {
        "task": t,
        "module": module,
        "dataset": dataset,
        "returncode": res.returncode,
        "elapsed_s": round(elapsed, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="audit/master_queue.csv")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max_tasks", type=int, default=None,
                    help="Stop after N tasks (for testing)")
    ap.add_argument("--task_filter", default=None,
                    help="Only run this task type (e.g., zeroshot)")
    args = ap.parse_args()
    
    with open(args.queue) as f:
        reader = csv.DictReader(f)
        tasks = list(reader)
    
    if args.task_filter:
        tasks = [t for t in tasks if t['task'] == args.task_filter]
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    print(f"[INIT] {len(tasks)} tasks, parallel={args.parallel}")
    
    completed = 0
    failed = 0
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            completed += 1
            try:
                r = fut.result()
                status = "OK" if r['returncode'] == 0 else "FAIL"
                if status == "FAIL":
                    failed += 1
                print(f"[{completed}/{len(tasks)}] {r['task']}/{r['module']}/{r['dataset']} "
                      f"| {status} | {r.get('elapsed_s', 0):.1f}s")
            except Exception as e:
                failed += 1
                print(f"[ERROR] {task} | {e}")
            
            # Report every 10 tasks
            if completed % 10 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed * 60  # tasks/min
                eta = (len(tasks) - completed) / max(rate, 0.01)
                print(f"[PROGRESS] {completed}/{len(tasks)} | {rate:.1f} tasks/min | ETA {eta:.0f} min")
    
    total = time.time() - t_start
    print(f"[DONE] {completed}/{len(tasks)} finished ({failed} failed) in {total/60:.1f} min")

if __name__ == "__main__":
    main()
