"""
Parallel zero-shot reasoning across many modules.
1 worker = 1 module (loads GPT-2 once, runs all 5 tasks).
Runs `parallel_workers` modules simultaneously on single GPU.
Idempotent: skips modules with existing complete output.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def is_complete(module_dir: Path) -> bool:
    """Check if all 5 task JSONs + detailed exist."""
    if not module_dir.exists():
        return False
    for t in TASKS:
        if not (module_dir / f"{t}.json").exists():
            return False
        if not (module_dir / f"{t}_detailed.json").exists():
            return False
    return True


def run_module(module: str, args) -> dict:
    out_dir = Path(args.output_base) / module
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    env = os.environ.copy()
    # Do not set HF_HUB_OFFLINE=1 because parquet datasets are accessed via resolve URLs
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)

    cmd = [
        sys.executable, "scripts/eval_zeroshot.py",
        "--module", module,
        "--model", args.model,
        "--tasks", ",".join(TASKS),
        "--num_samples", str(args.num_samples),
        "--batch_size", str(args.batch_size),
        "--output_dir", str(out_dir),
    ]

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    return {
        "module": module,
        "returncode": res.returncode,
        "elapsed_s": round(elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--modules_file", default="audit/zeroshot_modules.txt")
    ap.add_argument("--output_base", default="results/zeroshot_parallel")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    base_dir = Path(args.output_base)
    before = len(modules)
    modules = [m for m in modules if not is_complete(base_dir / m)]
    skipped = before - len(modules)
    if skipped > 0:
        print(f"[RESUME] Skipped {skipped} completed modules. {len(modules)} remaining.")

    print(f"[INIT] {len(modules)} modules to evaluate")
    print(f"[INIT] parallel = {args.parallel}")
    print(f"[INIT] ETA = {(len(modules)/max(1, args.parallel) * 120):.0f} seconds (~2 min/module)")
    print()

    t_start = time.time()
    completed = 0
    failed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_module, m, args): m for m in modules}
        for fut in as_completed(futures):
            module = futures[fut]
            try:
                r = fut.result()
                completed += 1
                status = "OK" if r.get("returncode") == 0 else f"FAIL({r['returncode']})"
                if r.get("returncode") != 0:
                    failed += 1
                    failures.append({
                        "module": module,
                        "returncode": r.get("returncode"),
                        "elapsed_s": r.get("elapsed_s"),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                print(f"[{completed}/{len(modules)}] {module} | {status} | {r.get('elapsed_s', 0):.1f}s")
            except Exception as e:
                failed += 1
                failures.append({
                    "module": module,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                print(f"[ERROR] {module} | {type(e).__name__}: {e}")

            # Write failures incrementally
            if failures:
                failures_log = Path("audit/zeroshot_failures.csv")
                failures_log.parent.mkdir(parents=True, exist_ok=True)
                with open(failures_log, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["module", "returncode", "error", "elapsed_s", "timestamp"])
                    w.writeheader()
                    for fail in failures:
                        w.writerow({k: fail.get(k, "") for k in w.fieldnames})

            # Auto-aggregate every 10 modules
            if completed % 10 == 0:
                print(f"[AGG] Updating audit/ at {completed} modules...")
                subprocess.run(
                    [sys.executable, "scripts/aggregate_zeroshot_details.py"],
                    cwd="D:/SVD_Project", check=False
                )

            # Git commit every 25 modules
            if completed % 25 == 0:
                print(f"[GIT] Committing at {completed} modules...")
                subprocess.run(["git", "add", "results/zeroshot_parallel/", "audit/zeroshot_*.csv", "audit/zeroshot_*.parquet", "audit/zeroshot_*.json"], cwd="D:/SVD_Project", check=False)
                subprocess.run(["git", "commit", "-m", f"feat(zeroshot): {completed} modules evaluated [auto]"], cwd="D:/SVD_Project", check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd="D:/SVD_Project", check=False)

            # Hugging Face upload every 50 modules
            if completed % 50 == 0:
                print(f"[HF] Uploading at {completed} modules...")
                subprocess.run([
                    "huggingface-cli", "upload",
                    "Hooshaai/svd-linear-attention-results",
                    "D:/SVD_Project/results/zeroshot_parallel",
                    "zeroshot_parallel",
                    "--repo-type", "dataset",
                ], check=False)

    total = time.time() - t_start
    print()
    print(f"[DONE] {completed} finished ({failed} failed) in {total:.1f}s")


if __name__ == "__main__":
    main()
