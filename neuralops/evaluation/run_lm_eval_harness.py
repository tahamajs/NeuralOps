#!/usr/bin/env python3
# Requires: pip install lm-eval
"""
LM Evaluation Harness Runner for Injected Attention and Compression Modules.

Evaluates GPT-2 downstream reasoning and QA capabilities across standard benchmarks:
  - HellaSwag: Commonsense reasoning
  - ARC-Easy & ARC-Challenge: Grade-school science questions
  - Winogrande: Adversarial pronoun disambiguation
  - GSM8K: Multi-step mathematical reasoning

Integrates with the framework module registry and apply_module_to_model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Lazy ML imports (imported in main to allow --help without heavy dependencies)
# ---------------------------------------------------------------------------
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
ATTENTION_REGISTRY = None
COMPRESSOR_REGISTRY = None
apply_module_to_model = None
HFLM = None
simple_evaluate = None
_HAS_LM_EVAL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("LMEvalHarness")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LM Evaluation Harness on GPT-2 with an injected attention or compressor module."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/gpt2_wikitext",
        help="Path or HF repo id of pretrained model (default: models/gpt2_wikitext)",
    )
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="Name of registered module (or 'full' for baseline)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Rank for low-rank approximation (default: 64)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hellaswag", "arc_easy", "arc_challenge", "winogrande", "gsm8k"],
        help="List of lm-evaluation-harness tasks to evaluate",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of samples per task (int or float <= 1.0, default: None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/06_lm_eval_harness",
        help="Directory to save evaluation results (default: results/06_lm_eval_harness)",
    )
    parser.add_argument(
        "--no_inject",
        action="store_true",
        help="If set, do not inject any module (evaluate raw baseline model)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="If set, run with limit=10 and tasks=['hellaswag'] only",
    )
    return parser.parse_args(argv)


def extract_task_metrics(
    raw_task_results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Extract key evaluation metrics (acc, acc_norm, exact_match, f1, etc.)
    from the raw lm_eval results structure.
    """
    extracted: dict[str, dict[str, float]] = {}
    for task_name, task_dict in raw_task_results.items():
        if not isinstance(task_dict, dict):
            continue
        metrics: dict[str, float] = {}
        for raw_k, val in task_dict.items():
            if not isinstance(val, (int, float)):
                continue
            # lm_eval metric keys frequently have commas, e.g., 'acc,none'
            clean_k = raw_k.split(",")[0].strip()
            # Prioritize standard metric families
            if any(term in clean_k for term in ["acc", "exact_match", "f1", "em", "norm"]):
                metrics[clean_k] = round(float(val), 4)

        if not metrics:
            # Fallback: record all scalar numbers if standard patterns were not matched
            for raw_k, val in task_dict.items():
                if isinstance(val, (int, float)):
                    clean_k = raw_k.split(",")[0].strip()
                    metrics[clean_k] = round(float(val), 4)

        extracted[task_name] = metrics
    return extracted


def print_summary_table(tasks_metrics: dict[str, dict[str, float]], module_name: str) -> None:
    """Print a clean Markdown-compatible summary table of results."""
    print("\n" + "=" * 65)
    print(f" LM EVALUATION HARNESS SUMMARY: {module_name}")
    print("=" * 65)
    header = f"{'Task':<22} | {'Metric':<18} | {'Value':<12}"
    print(header)
    print("-" * len(header))
    if not tasks_metrics:
        print(f"{'No metrics extracted or evaluation empty':<56}")
    else:
        for task_name, metrics in tasks_metrics.items():
            if not metrics:
                print(f"{task_name:<22} | {'(none)':<18} | {'N/A':<12}")
            else:
                for metric_k, val in metrics.items():
                    print(f"{task_name:<22} | {metric_k:<18} | {val:<12.4f}")
    print("=" * 65 + "\n")


def main(argv: list[str] | None = None) -> int:
    global \
        torch, \
        AutoModelForCausalLM, \
        AutoTokenizer, \
        ATTENTION_REGISTRY, \
        COMPRESSOR_REGISTRY, \
        apply_module_to_model, \
        HFLM, \
        simple_evaluate, \
        _HAS_LM_EVAL
    args = parse_args(argv)

    try:
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        _HAS_LM_EVAL = True
    except ImportError:
        _HAS_LM_EVAL = False

    if not _HAS_LM_EVAL:
        LOG.error("lm_eval is not installed. Please install it with: pip install lm-eval")
        print("pip install lm-eval", file=sys.stderr)
        return 1

    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        from neuralops.core.registry import ATTENTION_REGISTRY as _ATTENTION_REGISTRY
        from neuralops.core.registry import COMPRESSOR_REGISTRY as _COMPRESSOR_REGISTRY
        from neuralops.eval_all_models_datasets import (
            apply_module_to_model as _apply_module_to_model,
        )

        torch = _torch
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        ATTENTION_REGISTRY = _ATTENTION_REGISTRY
        COMPRESSOR_REGISTRY = _COMPRESSOR_REGISTRY
        apply_module_to_model = _apply_module_to_model
    except Exception as imp_err:
        LOG.error("Failed to import PyTorch or framework dependencies: %s", imp_err)
        return 1

    # Smoke test parameter adjustment
    if args.smoke_test:
        LOG.info("Smoke test enabled: limiting tasks to ['hellaswag'] and limit=10")
        args.tasks = ["hellaswag"]
        args.limit = 10

    if args.limit is not None and args.limit.is_integer():
        args.limit = int(args.limit)

    # Device verification
    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA requested but not available; falling back to CPU")
        device_str = "cpu"

    # Resolve model and tokenizer path
    tok_src = args.model_path if os.path.exists(args.model_path) else "gpt2"
    LOG.info("Loading tokenizer and model from '%s'...", tok_src)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(tok_src)

    # Inject module if requested
    injection_stats: dict[str, Any] = {}
    if not args.no_inject and args.module != "full":
        typ = "attention" if args.module in ATTENTION_REGISTRY else "compressor"
        LOG.info("Injecting %s module '%s' (rank=%d)...", typ, args.module, args.rank)
        try:
            model, stats = apply_module_to_model(
                model,
                args.module,
                module_type=typ,
                rank=args.rank,
                device=torch.device(device_str),
                tokenizer=tokenizer,
            )
            if isinstance(stats, dict):
                injection_stats = stats
            elif isinstance(stats, (int, float)):
                injection_stats = {"compression_rate": float(stats)}
            else:
                injection_stats = {"raw_stats": str(stats)}
        except Exception as e:
            LOG.error("Failed to inject module '%s': %s", args.module, e, exc_info=True)
            return 2
    else:
        LOG.info("Running baseline without injection (module='%s')", args.module)

    model.to(device_str)
    model.eval()

    # Wrap model into HFLM interface
    LOG.info(
        "Initializing lm-eval HFLM wrapper (batch_size=%d, device=%s)...",
        args.batch_size,
        device_str,
    )
    try:
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device_str,
        )
    except Exception as e:
        LOG.error("Failed to initialize HFLM wrapper: %s", e, exc_info=True)
        return 3

    # Execute simple_evaluate across tasks
    LOG.info("Starting lm_eval.simple_evaluate on tasks: %s", args.tasks)
    eval_kwargs: dict[str, Any] = {
        "model": lm,
        "tasks": args.tasks,
    }
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    eval_results: dict[str, Any] = {}
    try:
        eval_output = simple_evaluate(**eval_kwargs)
        eval_results = eval_output.get("results", {})
    except Exception as e:
        LOG.error("lm_eval evaluation error: %s", e, exc_info=True)
        # Attempt task-by-task fallback so partial results can be captured
        LOG.info("Attempting task-by-task execution fallback...")
        eval_results = {}
        for t in args.tasks:
            try:
                sub_kwargs = dict(eval_kwargs)
                sub_kwargs["tasks"] = [t]
                res = simple_evaluate(**sub_kwargs)
                eval_results.update(res.get("results", {}))
            except Exception as task_err:
                LOG.warning("Task '%s' failed in fallback: %s", t, task_err)
                eval_results[t] = {"error": str(task_err)}

    tasks_metrics = extract_task_metrics(eval_results)
    print_summary_table(tasks_metrics, args.module)

    # Save structured results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module}_lmeval.json"

    payload: dict[str, Any] = {
        "module": args.module,
        "model_path": args.model_path,
        "rank": args.rank,
        "num_fewshot": args.num_fewshot,
        "tasks": tasks_metrics,
        "raw_results": eval_results,
        "config": {
            "batch_size": args.batch_size,
            "limit": args.limit,
            "device": device_str,
            "no_inject": args.no_inject,
            "smoke_test": args.smoke_test,
            "injection_stats": injection_stats,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Saved results to %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
