#!/usr/bin/env python3
"""
Master Attention Benchmark Orchestrator
Executes Layer 0 (Sanity), Layer 1 (GLUE Evaluation), Layer 2 (Profiling),
Layer 3 (Stability), and Layer 4 (Diagnostics) across registered attention architectures.
"""

from __future__ import annotations
import os
import sys
import json
import yaml
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
CODES_DIR = PROJECT_ROOT / "Codes"

for p in [str(PROJECT_ROOT), str(BENCHMARK_DIR), str(CODES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# Framework and Benchmark imports
from benchmark.core.model_factory import build_model_with_attention
from benchmark.core.registry_utils import get_all_registered_models
from benchmark.tasks.dataset_loader import load_glue_dataset
from benchmark.tasks.glue_tasks import GLUE_TASK_CONFIGS
from benchmark.runners.sanity import run_sanity_check
from benchmark.runners.train import train_model
from benchmark.runners.profile import profile_model
from benchmark.runners.stability import evaluate_stability
from benchmark.diagnostics.attention_analysis import compute_attention_entropy, compute_attention_sparsity
from benchmark.diagnostics.moe_analysis import compute_moe_load_balance
from benchmark.diagnostics.gradient_analysis import compute_gradient_norm_stats
from benchmark.report_generator import generate_all_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure multi-sink logging to console and persistent log files."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Master file handler
    fh = logging.FileHandler(logs_dir / "benchmark.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Error file handler
    eh = logging.FileHandler(logs_dir / "errors.log")
    eh.setLevel(logging.WARNING)
    eh.setFormatter(formatter)
    logger.addHandler(eh)

    return logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Model Attention Benchmark Framework")
    parser.add_argument("--config", type=str, default="benchmark/configs/default.yaml", help="Path to default YAML config")
    parser.add_argument("--models-config", type=str, default="benchmark/configs/models_to_benchmark.yaml", help="Path to models list YAML")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model names to benchmark")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of GLUE tasks (e.g. 'sst2,mrpc')")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap train samples per task for rapid testing")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip Layer 2 efficiency profiling")
    parser.add_argument("--skip-stability", action="store_true", help="Skip Layer 3 multi-seed stability testing")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip Layer 4 structural diagnostics")
    parser.add_argument("--smoke-test", action="store_true", help="Run ultra-fast smoke test with synthetic subset")
    return parser.parse_args()


def load_configurations(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.models is not None:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    elif os.path.exists(args.models_config):
        with open(args.models_config, "r") as f:
            models_data = yaml.safe_load(f)
            model_list = models_data.get("models", ["full"])
    else:
        model_list = ["full", "svd_linear", "performer"]

    config["selected_models"] = model_list

    if args.tasks is not None:
        config["selected_tasks"] = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    else:
        config["selected_tasks"] = ["sst2", "mrpc"] if args.smoke_test else ["sst2", "mrpc", "qnli", "rte", "cola", "stsb"]

    if args.smoke_test:
        config["max_samples"] = 64
        config["profile_seq_lengths"] = [128, 256]
        config["profile_batch_sizes"] = [1, 8]
        config["seeds"] = [42]
        config["learning_rates"] = [3e-5]
    elif args.max_samples is not None:
        config["max_samples"] = args.max_samples

    return config


def main() -> None:
    args = parse_arguments()
    config = load_configurations(args)
    output_dir = Path(config.get("output_dir", "benchmark_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("🚀 INITIALIZING ATTENTION BENCHMARK FRAMEWORK")
    logger.info(f"Backbone: {config.get('model_name', 'distilbert-base-uncased')}")
    logger.info(f"Target Tasks: {config['selected_tasks']}")
    logger.info(f"Target Models ({len(config['selected_models'])}): {config['selected_models']}")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", "distilbert-base-uncased"))

    # Preload datasets
    logger.info("📦 Pre-loading and tokenizing GLUE datasets...")
    datasets_cache: Dict[str, Any] = {}
    for task in config["selected_tasks"]:
        try:
            train_ds, eval_ds, num_labels = load_glue_dataset(
                task_name=task,
                tokenizer=tokenizer,
                max_seq_len=int(config.get("max_seq_len", 128)),
                max_samples=config.get("max_samples"),
            )
            datasets_cache[task] = (train_ds, eval_ds, num_labels)
            logger.info(f"  ✓ Loaded task {task.upper()} (train: {len(train_ds)}, eval: {len(eval_ds)}, labels: {num_labels})")
        except Exception as e:
            logger.error(f"  ✗ Failed to load task {task}: {e}")

    results_file = output_dir / "results.json"
    results: Dict[str, Any] = {}
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for model_name in tqdm(config["selected_models"], desc="Benchmarking Models", unit="model"):
        logger.info("-" * 60)
        logger.info(f"🔍 Evaluating Candidate Architecture: [{model_name}]")
        logger.info("-" * 60)

        model_record: Dict[str, Any] = {
            "model_name": model_name,
            "status": "INITIALIZING",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "glue_tasks": {},
            "profiling": {},
            "stability": {},
            "diagnostics": {},
            "num_params": 0,
            "avg_glue": 0.0,
        }

        # -------------------------------------------------------------
        # STEP 1: MODEL BUILDING
        # -------------------------------------------------------------
        base_model, build_status = build_model_with_attention(
            model_name=model_name,
            base_model_name=config.get("model_name", "distilbert-base-uncased"),
            num_labels=2,
            pretrained=True,
        )

        if base_model is None or build_status != "OK":
            logger.warning(f"  ✗ Model instantiation failed: {build_status}")
            model_record["status"] = "FAIL_BUILD"
            model_record["error"] = build_status
            results[model_name] = model_record
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            continue

        model_record["num_params"] = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        logger.info(f"  ✓ Model built successfully ({model_record['num_params'] / 1e6:.2f}M trainable parameters)")

        # -------------------------------------------------------------
        # STEP 2: LAYER 0 - SANITY CHECK
        # -------------------------------------------------------------
        sanity_res = run_sanity_check(base_model, model_name)
        if not sanity_res["passed"]:
            logger.warning(f"  ✗ Layer 0 Sanity Check failed: {sanity_res['error']}")
            model_record["status"] = "FAIL_SANITY"
            model_record["error"] = sanity_res["error"]
            results[model_name] = model_record
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            continue
        logger.info("  ✓ Layer 0 Sanity Check: PASSED (Shapes, Dtypes, Non-NaN Gradients verified)")

        # -------------------------------------------------------------
        # STEP 3: LAYER 1 - GLUE TRAINING & EVALUATION
        # -------------------------------------------------------------
        task_scores = []
        for task in config["selected_tasks"]:
            if task not in datasets_cache:
                continue

            train_ds, eval_ds, num_labels = datasets_cache[task]

            # Re-instantiate model for task-specific classification head
            task_model, _ = build_model_with_attention(
                model_name=model_name,
                base_model_name=config.get("model_name", "distilbert-base-uncased"),
                num_labels=num_labels,
                pretrained=True,
            )

            train_res = train_model(
                model=task_model,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                task_name=task,
                config=config,
                seed=42,
                epochs=1 if args.smoke_test else None,
            )

            model_record["glue_tasks"][task] = train_res
            t_score = train_res.get("eval_metrics", {}).get("score", train_res.get("eval_metrics", {}).get("accuracy", 0.0))
            task_scores.append(t_score)
            logger.info(f"  ✓ Task {task.upper()} ({train_res['status']}): Score = {t_score:.4f} (Time: {train_res['training_time']:.1f}s)")

        model_record["avg_glue"] = float(sum(task_scores) / max(len(task_scores), 1))
        logger.info(f"  ★ Average GLUE Score: {model_record['avg_glue']:.4f}")

        # -------------------------------------------------------------
        # STEP 4: LAYER 2 - EFFICIENCY PROFILING
        # -------------------------------------------------------------
        if not args.skip_profiling:
            logger.info("  ⚡ Running Layer 2 Efficiency & Scaling Profiling...")
            prof_model, _ = build_model_with_attention(model_name=model_name, num_labels=2, pretrained=False)
            prof_res = profile_model(
                model=prof_model,
                seq_lengths=config.get("profile_seq_lengths", [128, 256, 512]),
                batch_sizes=config.get("profile_batch_sizes", [1, 32]),
                num_warmup=2 if args.smoke_test else 5,
                num_iterations=10 if args.smoke_test else 50,
            )
            model_record["profiling"] = prof_res
            cfg_512 = prof_res.get("configurations", {}).get("bs32_seq512", prof_res.get("configurations", {}).get("bs8_seq256", {}))
            logger.info(f"  ✓ Throughput: {cfg_512.get('tokens_per_sec', 0):,.1f} tok/s | Peak VRAM: {cfg_512.get('peak_vram_mb', 'N/A')}")

        # -------------------------------------------------------------
        # STEP 5: LAYER 3 - STABILITY EVALUATION
        # -------------------------------------------------------------
        if not args.skip_stability and "sst2" in datasets_cache:
            logger.info("  🛡️ Running Layer 3 Multi-Seed & LR Stability Audit...")
            train_sst2, eval_sst2, _ = datasets_cache["sst2"]
            stab_res = evaluate_stability(
                model_builder_fn=lambda name: build_model_with_attention(name, num_labels=2, pretrained=True),
                model_name=model_name,
                config=config,
                train_dataset=train_sst2,
                eval_dataset=eval_sst2,
                task_name="sst2",
            )
            model_record["stability"] = stab_res
            logger.info(f"  ✓ Stability Verdict: {'STABLE' if stab_res['stable'] else 'UNSTABLE'} (Mean Acc: {stab_res['mean_seed_acc']:.4f}, CV_LR: {stab_res['cv_lr']:.3f})")

        # -------------------------------------------------------------
        # STEP 6: LAYER 4 - STRUCTURAL DIAGNOSTICS
        # -------------------------------------------------------------
        if not args.skip_diagnostics:
            logger.info("  🔬 Running Layer 4 Structural & Gradient Diagnostics...")
            diag_model, _ = build_model_with_attention(model_name=model_name, num_labels=2, pretrained=True)
            dummy_inputs = {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "attention_mask": torch.ones((4, 64)),
            }
            grad_stats = compute_gradient_norm_stats(diag_model, dummy_inputs)
            moe_stats = compute_moe_load_balance(diag_model, dummy_inputs)

            model_record["diagnostics"] = {
                "gradient_norm": grad_stats,
                "moe_balance": moe_stats,
            }
            logger.info(f"  ✓ Diagnostics: Mean Grad Norm = {grad_stats['mean_norm']:.4f}, NaN = {grad_stats['has_nan']}")

        model_record["status"] = "OK"
        results[model_name] = model_record

        # Incrementally persist results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # -----------------------------------------------------------------
    # FINAL REPORT & LEADERBOARD GENERATION
    # -----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("📊 GENERATING BENCHMARK REPORTS & VISUALIZATION ARTIFACTS")
    logger.info("=" * 70)

    generate_all_reports(str(results_file), str(output_dir))
    logger.info(f"✓ Leaderboard and summary exported to {output_dir}")


if __name__ == "__main__":
    main()
