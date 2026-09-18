#!/usr/bin/env python3
"""
Child runner: executes exactly one module evaluation.
Called by run_quality_sweep.py via subprocess.Popen.
Writes result JSON to --output_json path and exits with code 0 or 1.
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
from datetime import datetime


def main():
    from framework.run_quality_sweep import parse_args, run_single_module_main

    parser_custom = argparse.ArgumentParser(add_help=False)
    parser_custom.add_argument("--module", required=True)
    parser_custom.add_argument("--output_json", required=True)
    known, remaining = parser_custom.parse_known_args()

    args = parse_args(remaining)
    args.module = known.module
    args.output_json = known.output_json
    args.modules = [known.module]

    result = {
        "module": args.module,
        "model": args.model,
        "dataset": args.dataset,
        "status": "UNKNOWN",
        "started_at": datetime.utcnow().isoformat(),
        "pid": __import__("os").getpid(),
    }

    try:
        # Late import so the process can be killed if imports take too long
        from framework.run_quality_sweep import run_single_module_main

        # Call the single-module evaluation as a library function
        metrics = run_single_module_main(args)
        result.update({
            "status": metrics.get("status", "PASS"),
            "accuracy": metrics.get("accuracy_pct"),
            "f1": metrics.get("f1_score"),
            "params_before": metrics.get("total_params_before"),
            "params_after": metrics.get("total_params_after"),
            "compression_ratio": metrics.get("compression_rate"),
            "peak_vram_mb": metrics.get("peak_vram_mb"),
            "elapsed_s": metrics.get("elapsed_s"),
            "error": "",
        })
        exit_code = 0
    except Exception as e:
        result.update({
            "status": "CRASH",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[:5000],
        })
        exit_code = 1

    result["finished_at"] = datetime.utcnow().isoformat()
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
