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
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_json", required=True,
                    help="Path to write the result JSON")
    ap.add_argument("--output_dir", required=True,
                    help="Directory for checkpoints and metrics")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--recovery_steps", type=int, default=50)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout_hint", type=int, default=600)
    
    # Optional arguments that might be passed from the parent sweep script
    ap.add_argument("--optimizer", type=str, default="adamw")
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--use_muon", action="store_true")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--finetune_mode", type=str, default="lora")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--target_layer", type=str, default="all")

    args = ap.parse_args()
    args.modules = [args.module]

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
