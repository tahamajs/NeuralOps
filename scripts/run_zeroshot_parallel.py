"""
Parallel zero-shot reasoning across many modules.
1 worker = 1 module (loads GPT-2 once, runs all 5 tasks).
Runs `parallel_workers` modules simultaneously on single GPU.
Idempotent: skips modules with existing complete output.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def is_complete(module: str, base_dir: Path) -> bool:
    """Check if all 5 tasks have output for this module."""
    mdir = base_dir / module
    if not mdir.exists():
        return False
    return all((mdir / f"{t}.json").exists() for t in TASKS)


def run_module(module: str, args) -> dict:
    out_dir = Path(args.output_base) / module
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    env = os.environ.copy()
    # Allow online access for parquet downloads if not cached
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)

    cmd = [
        sys.executable, "scripts/eval_zeroshot.py",
        "--module", module,
        "--model", args.model,
        "--tasks", ",".join(TASKS),
        "--num_samples", str(args.num_samples),
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
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--modules_file", default="audit/zeroshot_modules.txt")
    ap.add_argument("--output_base", default="results/zeroshot_parallel")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if args.resume:
        base = Path(args.output_base)
        modules = [m for m in modules if not is_complete(m, base)]

    print(f"[INIT] {len(modules)} modules to evaluate")
    print(f"[INIT] parallel = {args.parallel}")
    print(f"[INIT] ETA = {(len(modules)/max(1, args.parallel) * 120):.0f} seconds (~2 min/module)")
    print()

    t_start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_module, m, args): m for m in modules}
        for fut in as_completed(futures):
            module = futures[fut]
            try:
                r = fut.result()
                completed += 1
                status = "OK" if r["returncode"] == 0 else f"FAIL({r['returncode']})"
                if r["returncode"] != 0:
                    failed += 1
                print(f"[{completed}/{len(modules)}] {module} | {status} | {r['elapsed_s']:.1f}s")
            except Exception as e:
                failed += 1
                print(f"[ERROR] {module} | {e}")

    total = time.time() - t_start
    print()
    print(f"[DONE] {completed} finished ({failed} failed) in {total:.1f}s")


if __name__ == "__main__":
    main()
