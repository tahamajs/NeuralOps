"""
Main CLI entrypoint and orchestrator for the Attention Benchmark Suite.
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = PROJECT_ROOT / "Codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.core.model_factory import build_model_with_attention
from benchmark.core.seed import set_seed
from benchmark.tasks.glue_tasks import load_and_tokenize_glue_task
from benchmark.runners.sanity import run_sanity_check
from benchmark.runners.train import train_model
from benchmark.runners.profile import profile_model
from benchmark.runners.stability import evaluate_stability
from benchmark.diagnostics.gradient_analysis import compute_gradient_norm_stats
from benchmark.diagnostics.moe_analysis import compute_moe_load_balance
from benchmark.report_generator import generate_all_reports

logger = logging.getLogger("benchmark")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "logs" / "benchmark.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), mode="a"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Layer Attention Benchmark Suite")
    parser.add_argument("--config", type=str, default="benchmark/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model names to benchmark")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of GLUE tasks to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save benchmark results")
    parser.add_argument("--smoke-test", action="store_true", help="Run in fast smoke test mode (1 epoch, few steps)")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip Layer 2 profiling")
    parser.add_argument("--skip-stability", action="store_true", help="Skip Layer 3 stability audit")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip Layer 4 diagnostics")
    parser.add_argument("--report-only", type=str, default=None, help="Generate report from existing results.json")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if p.exists():
        with open(p, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    args = parse_args()

    if args.report_only:
        res_p = Path(args.report_only)
        out_p = res_p.parent
        generate_all_reports(str(res_p), str(out_p))
        print(f"✓ Reports generated successfully in {out_p}")
        return

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir

    output_dir = Path(config.get("output_dir", "benchmark_output"))
    setup_logging(output_dir)

    # Setup Target Execution Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Target Device: CUDA ({torch.cuda.get_device_name()})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Target Device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Target Device: CPU")

    set_seed(config.get("seed", 42))

    if args.models:
        config["selected_models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    elif "selected_models" not in config:
        config["selected_models"] = ["full", "svd_linear"]

    if args.tasks:
        config["selected_tasks"] = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    elif "selected_tasks" not in config:
        config["selected_tasks"] = ["sst2", "mrpc"]

    if args.smoke_test:
        config["smoke_test"] = True
        config["selected_tasks"] = config["selected_tasks"][:2]
        config["max_train_samples"] = 64
        config["max_eval_samples"] = 32

    logger.info("=" * 70)
    logger.info("🚀 INITIALIZING ATTENTION BENCHMARK FRAMEWORK")
    logger.info(f"Backbone: {config.get('model_name', 'distilbert-base-uncased')}")
    logger.info(f"Target Tasks: {config['selected_tasks']}")
    logger.info(f"Target Models ({len(config['selected_models'])}): {config['selected_models']}")
    logger.info("=" * 70)

    # Preload and tokenize GLUE datasets
    logger.info("📦 Pre-loading and tokenizing GLUE datasets...")
    datasets_cache: Dict[str, Any] = {}
    for task in config["selected_tasks"]:
        try:
            train_ds, eval_ds, num_labels = load_and_tokenize_glue_task(
                task_name=task,
                tokenizer_name=config.get("model_name", "distilbert-base-uncased"),
                max_seq_length=config.get("max_seq_length", 128),
                max_train_samples=config.get("max_train_samples", None),
                max_eval_samples=config.get("max_eval_samples", None),
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

        base_model = base_model.to(device)
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
            if task_model is not None:
                task_model = task_model.to(device)

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
            if prof_model is not None:
                prof_model = prof_model.to(device)
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
            if diag_model is not None:
                diag_model = diag_model.to(device)
            dummy_inputs = {
                "input_ids": torch.randint(0, 1000, (4, 64), device=device),
                "attention_mask": torch.ones((4, 64), device=device),
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
