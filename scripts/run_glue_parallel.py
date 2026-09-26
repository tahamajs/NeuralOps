"""
Automated parallel GLUE / Classification quality sweep runner across all framework modules.
Runs 4 workers in parallel with process isolation on the GPU server.
Safely checks for existing results in summary.csv to allow resuming without duplicate computation.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

ATTN_DIR = PROJECT_ROOT / "Codes" / "framework" / "attentions"
COMP_DIR = PROJECT_ROOT / "Codes" / "framework" / "compression_engine"


def get_all_modules():
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "Codes"))
        from framework.run_quality_sweep import modules
        return sorted(list(modules()))
    except Exception as e:
        print(f"Warning: Failed to load canonical modules(): {e}, falling back to directory scan", flush=True)
        mods = set()
        for d in [ATTN_DIR, COMP_DIR]:
            if d.exists():
                for p in d.glob("*.py"):
                    name = p.name
                    if not name.startswith("__") and not name.startswith("._") and not name.startswith("."):
                        mods.add(p.stem)
        return sorted(list(mods))


def get_existing_completed(output_dir: Path):
    summary_file = output_dir / "summary.csv"
    if not summary_file.exists():
        return set()
    completed = set()
    try:
        with open(summary_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                mod = r.get("module_name")
                status = r.get("status", "")
                if mod and status in {
                    "PASS", "PASS_LOW", "REJECTED_EXPANSION",
                    "EXTREME_EXPANSION", "MODERATE_EXPANSION",
                    "TIMEOUT", "SKIP_BERT_ARCHITECTURAL", "SKIP_BLOCKED",
                }:
                    completed.add(mod)
    except Exception:
        pass
    return completed


def run_single_module(module: str, args, output_dir: Path):
    # Wait until system has healthy RAM headroom (>= 2.0 GB)
    while True:
        free_ram_gb = psutil.virtual_memory().available / 1e9
        if free_ram_gb >= 2.0:
            break
        time.sleep(2)

    cmd = [
        sys.executable,
        "Codes/framework/run_quality_sweep.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--dataset_dir", args.dataset_dir,
        "--modules", module,
        "--output_dir", str(output_dir),
        "--batch_size", str(args.batch_size),
        "--parallel", "1",
        "--recovery_steps", str(args.recovery_steps),
        "--timeout", str(args.timeout),
    ]

    t0 = time.time()
    try:
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - t0
        return {
            "module": module,
            "returncode": res.returncode,
            "elapsed_s": round(elapsed, 2),
            "output": res.stdout[-1000:] if res.stdout else "",
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "module": module,
            "returncode": -1,
            "elapsed_s": round(elapsed, 2),
            "output": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run parallel GLUE sweeps across modules")
    parser.add_argument("--model", default="models/distilbert_sst2", help="Pretrained model path")
    parser.add_argument("--dataset", default="sst2", help="GLUE dataset name")
    parser.add_argument("--dataset_dir", default="Codes/framework/data/glue_data/sst2", help="Path to tokenized dataset")
    parser.add_argument("--output_dir", default="results/02_glue_sweeps/distilbert/sst2_4parallel", help="Output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per worker")
    parser.add_argument("--recovery_steps", type=int, default=50, help="Fine-tuning recovery steps")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-module timeout in seconds")
    parser.add_argument("--modules", default=None, help="Comma-separated modules list or path to txt file")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.modules:
        if Path(args.modules).exists():
            modules = [l.strip() for l in Path(args.modules).read_text().splitlines() if l.strip() and not l.startswith("#")]
        else:
            modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    else:
        modules = get_all_modules()

    done = get_existing_completed(out_dir)
    pending = [m for m in modules if m not in done]

    print(f"================================================================================", flush=True)
    print(f"PARALLEL GLUE SWEEP ORCHESTRATOR", flush=True)
    print(f"  Model:           {args.model}", flush=True)
    print(f"  Dataset:         {args.dataset} ({args.dataset_dir})", flush=True)
    print(f"  Output Dir:      {out_dir}", flush=True)
    print(f"  Concurrency:     {args.parallel} parallel workers", flush=True)
    print(f"  Batch Size:      {args.batch_size} (Recovery: {args.recovery_steps} steps)", flush=True)
    print(f"  Total Modules:   {len(modules)}", flush=True)
    print(f"  Already Done:    {len(done)}", flush=True)
    print(f"  Pending:         {len(pending)}", flush=True)
    print(f"================================================================================", flush=True)

    if not pending:
        print("All requested modules are already evaluated! Exiting.", flush=True)
        return

    completed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_single_module, m, args, out_dir): m for m in pending}
        for future in as_completed(futures):
            mod = futures[future]
            try:
                res = future.result()
                completed_count += 1
                if res["returncode"] == 0:
                    status_str = "SUCCESS"
                    print(f"[{completed_count}/{len(pending)}] {mod:35} | {status_str:10} | {res['elapsed_s']:6.1f}s", flush=True)
                else:
                    status_str = f"EXIT({res['returncode']})"
                    failed_count += 1
                    print(f"[{completed_count}/{len(pending)}] {mod:35} | {status_str:10} | {res['elapsed_s']:6.1f}s | Output: {res['output'][-200:] if res['output'] else 'none'}", flush=True)
            except Exception as e:
                completed_count += 1
                failed_count += 1
                print(f"[{completed_count}/{len(pending)}] {mod:35} | EXCEPTION: {e}", flush=True)


if __name__ == "__main__":
    main()
