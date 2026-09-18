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
    # Never let the child read a stale summary.csv — always run fresh
    args.fresh = True

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
            # Status and accuracy
            "status":                       metrics.get("status", "PASS"),
            "accuracy":                     metrics.get("accuracy_pct", 0.0),
            "accuracy_pct":                 metrics.get("accuracy_pct", 0.0),
            # F1 / classification metrics
            "f1":                           metrics.get("f1_score", 0.0),
            "f1_score":                     metrics.get("f1_score", 0.0),
            "f1_macro":                     metrics.get("f1_macro", 0.0),
            "precision":                    metrics.get("precision", 0.0),
            "recall":                       metrics.get("recall", 0.0),
            "val_loss":                     metrics.get("val_loss", 0.0),
            # Speed / VRAM
            "peak_vram_mb":                 metrics.get("peak_vram_mb", 0.0),
            "tokens_per_sec":               metrics.get("tokens_per_sec", 0.0),
            "elapsed_s":                    metrics.get("elapsed_s", 0.0),
            # Compression / param counts
            "params_before":                metrics.get("total_params_before", 0),
            "params_after":                 metrics.get("total_params_after", 0),
            "total_params_before":          metrics.get("total_params_before", 0),
            "total_params_after":           metrics.get("total_params_after", 0),
            "trainable_recovery_params":    metrics.get("trainable_recovery_params", 0),
            "compression_ratio":            metrics.get("compression_rate", 1.0),
            "compression_rate":             metrics.get("compression_rate", 1.0),
            "compression_ratio_pct":        metrics.get("compression_ratio_pct", 0.0),
            "flops_reduction_pct":          metrics.get("flops_reduction_pct", 0.0),
            "num_compressed_layers":        metrics.get("num_compressed_layers", 0),
            "num_fallback_layers":          metrics.get("num_fallback_layers", 0),
            "avg_reconstruction_error":     metrics.get("avg_reconstruction_error", ""),
            "weighted_reconstruction_error":metrics.get("weighted_reconstruction_error", ""),
            "num_injected_attention_layers":metrics.get("num_injected_attention_layers", 0),
            "lora_params":                  metrics.get("lora_params", 0),
            # Path
            "path":                         metrics.get("path", ""),
            "error":                        "",
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
