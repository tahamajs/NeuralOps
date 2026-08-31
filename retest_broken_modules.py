#!/usr/bin/env python3
"""
Retest Broken Modules Benchmark Runner.
Evaluates fixed attention and compression modules from broken_modules.txt on DistilBERT / SST-2.
Enforces strict 300s (5-minute) timeout and saves results to results/run_broken_fix.
"""

import sys
from pathlib import Path

# Add project root and Codes directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

from framework.run_quality_sweep import run_quality_benchmark_sweep

def main():
    broken_txt = PROJECT_DIR / "broken_modules.txt"
    if not broken_txt.exists():
        print(f"Error: {broken_txt} not found.")
        sys.exit(1)

    with open(broken_txt, "r", encoding="utf-8") as f:
        modules = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # Clean modules: exclude injection helpers
    modules = [m for m in modules if not (m.endswith("_inject") or m.endswith("_injection") or m.endswith("_fn"))]
    print(f"Loaded {len(modules)} broken modules to re-evaluate: {modules}")

    output_dir = PROJECT_DIR / "results" / "run_broken_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_quality_benchmark_sweep(
        modules=modules,
        models=["distilbert"],
        datasets=["sst2"],
        seeds=[42],
        epochs=1,
        max_steps=100,
        batch_size=128,
        seq_len=128,
        timeout=300,
        precision="fp16",
        output_dir=str(output_dir),
        pretrained=True,
        gradient_checkpointing=True,
    )

    print(f"\nRetest Complete! Run ID: {run_id}")
    print(f"Results saved to: {output_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()
