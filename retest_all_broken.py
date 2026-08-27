#!/usr/bin/env python3
"""
Retest script for all genuine broken modules identified during Phase 2 screening.
Enforces strict 5-minute (300s) timeout per module and saves results to results/run_all_broken_fix.
"""

import sys
import subprocess
from pathlib import Path

def main():
    project_dir = Path(__file__).resolve().parent
    broken_modules_file = project_dir / "broken_modules.txt"
    output_dir = project_dir / "results" / "run_all_broken_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not broken_modules_file.exists():
        print(f"Error: {broken_modules_file} not found.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-u",
        str(project_dir / "Codes" / "framework" / "run_quality_sweep.py"),
        "--modules", str(broken_modules_file),
        "--models", "distilbert",
        "--datasets", "sst2",
        "--seeds", "42",
        "--epochs", "1",
        "--max_steps", "100",
        "--batch_size", "128",
        "--seq_len", "128",
        "--timeout", "300",
        "--precision", "fp16",
        "--pretrained",
        "--gradient_checkpointing",
        "--output_dir", str(output_dir),
    ]

    print("Launching broken modules retest with 300s timeout...")
    print(f"Command: {' '.join(cmd)}")
    sys.stdout.flush()

    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
