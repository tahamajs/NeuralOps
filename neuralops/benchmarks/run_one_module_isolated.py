#!/usr/bin/env python3
"""
Child runner: executes exactly one module evaluation.
Called by run_quality_sweep.py via subprocess.Popen.
Writes result JSON to --output_json path and exits with code 0 or 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT), str(ROOT / "Codes")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    from neuralops.run_quality_sweep import parse_args, run_single_module_main

    parser_custom = argparse.ArgumentParser(add_help=False)
    parser_custom.add_argument("--module", required=True)
    parser_custom.add_argument("--output_json", required=True)
    parser_custom.add_argument(
        "--perplexity",
        action="store_true",
        default=False,
        help="Evaluate causal language modeling perplexity on WikiText-2",
    )
    parser_custom.add_argument(
        "--long_context",
        action="store_true",
        default=False,
        help="Evaluate 4096-token sequence classification on Byte-Level IMDb",
    )
    known, remaining = parser_custom.parse_known_args()

    args = parse_args(remaining)
    args.module = known.module
    args.output_json = known.output_json
    args.perplexity = getattr(known, "perplexity", False)
    args.long_context = getattr(known, "long_context", False)
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
        from neuralops.run_quality_sweep import (
            DEPRECATED_MODULES,
            run_single_module_main,
        )

        if args.module in DEPRECATED_MODULES:
            result.update(
                {
                    "status": "SKIP_DEPRECATED",
                    "error": f"Deprecated module: {DEPRECATED_MODULES[args.module]}",
                    "finished_at": datetime.utcnow().isoformat(),
                }
            )
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            sys.exit(0)

        # CRITICAL: force parallel=1 so run_single_module_main uses the inline
        # evaluation path (_execute_single directly), NOT the ThreadPoolExecutor
        # path which would spawn another child subprocess recursively.
        args.parallel = 1

        # Call the single-module evaluation as a library function
        metrics = run_single_module_main(args)
        if isinstance(metrics, dict) and args.module in metrics:
            metrics = metrics[args.module]

        # Check parameter expansion — exclude LoRA adapter params from base model count
        p_before = int(metrics.get("total_params_before", 0) or 0)
        p_after = int(metrics.get("total_params_after", 0) or 0)
        lora_p = int(metrics.get("lora_params", 0) or 0)
        base_params_after = p_after - lora_p
        status = metrics.get("status", "PASS")
        error_msg = metrics.get("error", "") or ""

        if p_before > 0 and args.module != "full" and status in ("PASS", "PASS_LOW"):
            base_delta = base_params_after - p_before
            base_delta_pct = 100.0 * base_delta / p_before if p_before > 0 else 0.0
            if base_params_after > p_before * 1.20:
                status = "EXTREME_EXPANSION"
                expansion_err = (
                    f"EXTREME_EXPANSION: base_before={p_before} base_after={base_params_after} "
                    f"base_delta={base_delta} (+{base_delta_pct:.1f}%) lora_excluded={lora_p}"
                )
                error_msg = f"{error_msg}; {expansion_err}".strip("; ")
            elif (
                base_params_after > p_before
                and base_delta > 1024
                and base_delta_pct > 0.05
            ):
                status = "MODERATE_EXPANSION"
                expansion_err = (
                    f"MODERATE_EXPANSION: base_before={p_before} base_after={base_params_after} "
                    f"base_delta={base_delta} (+{base_delta_pct:.2f}%) lora_excluded={lora_p}"
                )
                error_msg = f"{error_msg}; {expansion_err}".strip("; ")

        acc_val = float(metrics.get("accuracy_pct", 0.0) or 0.0)
        elapsed_val = float(metrics.get("elapsed_s", 0.0) or 0.0)
        is_lm = (
            "wikitext" in str(getattr(args, "dataset", "")).lower()
            or str(getattr(args, "dataset", "")).lower() == "c4"
            or getattr(args, "task_type", "") == "causal_lm"
        )
        if (
            not is_lm
            and status in ("PASS", "PASS_LOW")
            and (acc_val <= 0.0 or elapsed_val < 0.5)
        ):
            status = "EVAL_FAILED"
            fail_err = f"EVAL_FAILED: invalid evaluation results (acc={acc_val:.2f}%, elapsed={elapsed_val:.2f}s)"
            error_msg = f"{error_msg}; {fail_err}".strip("; ")

        # metrics is the row dict for this module with all fields populated
        result.update(
            {
                # Status and accuracy
                "status": status,
                "accuracy": float(metrics.get("accuracy_pct", 0.0) or 0.0),
                "accuracy_pct": float(metrics.get("accuracy_pct", 0.0) or 0.0),
                # F1 / classification metrics
                "f1": float(metrics.get("f1_score", 0.0) or 0.0),
                "f1_score": float(metrics.get("f1_score", 0.0) or 0.0),
                "f1_macro": float(metrics.get("f1_macro", 0.0) or 0.0),
                "precision": float(metrics.get("precision", 0.0) or 0.0),
                "recall": float(metrics.get("recall", 0.0) or 0.0),
                "val_loss": float(metrics.get("val_loss", 0.0) or 0.0),
                # Speed / VRAM / Pareto metrics
                "peak_vram_mb": float(metrics.get("peak_vram_mb", 0.0) or 0.0),
                "tokens_per_sec": float(metrics.get("tokens_per_sec", 0.0) or 0.0),
                "throughput": float(metrics.get("tokens_per_sec", 0.0) or 0.0),
                "throughput_tok_per_sec": float(
                    metrics.get("tokens_per_sec", 0.0) or 0.0
                ),
                "latency_ms_per_token": float(
                    metrics.get("latency_ms_per_token", metrics.get("latency_ms", 0.0))
                    or 0.0
                ),
                "elapsed_s": float(metrics.get("elapsed_s", 0.0) or 0.0),
                "perplexity": (
                    float(metrics.get("perplexity", 0.0) or 0.0)
                    if metrics.get("perplexity") is not None
                    else ""
                ),
                "long_context_accuracy": (
                    float(metrics.get("long_context_accuracy", 0.0) or 0.0)
                    if metrics.get("long_context_accuracy") is not None
                    else ""
                ),
                # Compression / param counts
                "params_before": p_before,
                "params_after": p_after,
                "total_params_before": p_before,
                "total_params_after": p_after,
                "trainable_recovery_params": int(
                    metrics.get("trainable_recovery_params", 0) or 0
                ),
                "compression_ratio": float(metrics.get("compression_rate", 1.0) or 1.0),
                "compression_rate": float(metrics.get("compression_rate", 1.0) or 1.0),
                "compression_rate_raw": float(
                    metrics.get("compression_rate_raw", 1.0) or 1.0
                ),
                "compression_rate_net": float(
                    metrics.get("compression_rate_net", 1.0) or 1.0
                ),
                "compression_ratio_pct": float(
                    metrics.get("compression_ratio_pct", 0.0) or 0.0
                ),
                "flops_reduction_pct": float(
                    metrics.get("flops_reduction_pct", 0.0) or 0.0
                ),
                "num_compressed_layers": int(
                    metrics.get("num_compressed_layers", 0) or 0
                ),
                "num_fallback_layers": int(metrics.get("num_fallback_layers", 0) or 0),
                "avg_reconstruction_error": metrics.get("avg_reconstruction_error", "")
                or "",
                "weighted_reconstruction_error": metrics.get(
                    "weighted_reconstruction_error", ""
                )
                or "",
                "num_injected_attention_layers": int(
                    metrics.get("num_injected_attention_layers", 0) or 0
                ),
                "lora_params": int(metrics.get("lora_params", 0) or 0),
                "calibration_time_s": float(
                    metrics.get("calibration_time_s", 0.0) or 0.0
                ),
                "recovery_time_s": float(metrics.get("recovery_time_s", 0.0) or 0.0),
                # Path
                "path": metrics.get("path", "") or "",
                "error": error_msg,
            }
        )
        exit_code = 0
    except Exception as e:
        result.update(
            {
                "status": "CRASH",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:5000],
            }
        )
        exit_code = 1

    result["finished_at"] = datetime.utcnow().isoformat()
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
