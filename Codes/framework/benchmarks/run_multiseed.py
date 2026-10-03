#!/usr/bin/env python3
"""
Multi-Seed Evaluation Runner: Measures variance and statistical confidence (mean +/- std).

Evaluates key methods across 3 deterministic random seeds: [42, 123, 456]
Dataset: MNLI (Matched validation set)
Target Rank: 64

Outputs:
  - results/multiseed/{seed}/{module}.json
  - results/multiseed/final_summary.csv with columns: module, mean_acc, std_acc, mean_f1, std_f1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, load_dataset

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model
from framework.run_quality_sweep import prepare_eval_batches, evaluate_comprehensive

LOG = logging.getLogger("MultiSeedRunner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEEDS = [42, 123, 456]
TARGET_MODULES = ["dense", "a3", "kmeans", "aasvd", "adaptive_svd", "adaptive"]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed statistical confidence benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="models/distilbert_mnli_full",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="Codes/framework/data/glue_data/mnli",
                        help="Path to MNLI dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/06_multiseed/multiseed",
                        help="Directory to save multi-seed results")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="List or comma-separated string of random seeds")
    parser.add_argument("--modules", type=str, default="dense,a3,kmeans,aasvd,adaptive_svd,adaptive",
                        help="List or comma-separated string of modules to evaluate")
    parser.add_argument("--rank", type=int, default=64,
                        help="Target rank for low-rank decomposition")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for validation (auto-sizes to 32/16/8 based on VRAM if not specified)")
    parser.add_argument("--dataset", type=str, default="mnli",
                        help="Target dataset (default: mnli)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel evaluations (1 for deterministic single GPU)")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print planned evaluation matrix without running")
    args = parser.parse_args()

    if isinstance(args.seeds, str):
        args.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if isinstance(args.modules, str):
        args.modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    # Map full to dense if passed
    args.modules = ["dense" if m == "full" else m for m in args.modules]
    return args


def main():
    args = parse_args()
    initial_seed = args.seeds[0] if args.seeds else 42
    seed_everything(initial_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.batch_size is None:
        if device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                args.batch_size = 32 if free_gb >= 6.0 else (16 if free_gb >= 4.0 else 8)
            except Exception:
                args.batch_size = 16
        else:
            args.batch_size = 16

    print("=" * 80)
    print("MULTI-SEED STATISTICAL RIGOR EVALUATION (MNLI)")
    print(f"Seeds:      {args.seeds}")
    print(f"Modules:    {args.modules}")
    print(f"Rank:       {args.rank}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Planned multi-seed trials:")
        for seed in args.seeds:
            for mod in args.modules:
                print(f"  Seed: {seed} | Module: {mod:<15} | Out: {args.output_dir}/{seed}/{mod}.json")
        print("\n[DRY RUN COMPLETE] No sweeps were executed.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "final_summary.csv"

    # Load dataset validation split
    if Path(args.dataset_dir).exists():
        ds = load_from_disk(args.dataset_dir)
    else:
        ds = load_dataset("glue", "mnli")
    val_split = ds["validation_matched"] if "validation_matched" in ds else ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eval_batches = prepare_eval_batches(tokenizer, val_split, "mnli", 128, args.batch_size, device)
    LOG.info("Prepared %d validation batches for MNLI evaluation", len(eval_batches))

    # Store runs: module -> list of accuracies and F1s
    module_runs = {m: {"acc": [], "f1": []} for m in args.modules}

    for seed in args.seeds:
        seed_dir = out_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        LOG.info("=== Running Multi-Seed Evaluations for Seed %d ===", seed)

        for mod in args.modules:
            LOG.info("[Seed %d] Evaluating module: %s...", seed, mod)
            res_file = seed_dir / f"{mod}.json"
            t0 = time.time()

            try:
                model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
                model.eval()
                model.to(device)

                if mod != "dense":
                    typ = "attention" if mod in ATTENTION_REGISTRY else "compressor"
                    model, comp_stats = apply_module_to_model(
                        model, mod, typ, rank=args.rank, device=device
                    )
                    model.eval()
                    model.to(device)

                eval_res = evaluate_comprehensive(
                    model, tokenizer, val_split, "mnli", 128, args.batch_size, device,
                    fp16=True, precomputed_batches=eval_batches
                )
                acc = float(eval_res["accuracy_pct"])
                f1 = float(eval_res["f1_score"])

                module_runs[mod]["acc"].append(acc)
                module_runs[mod]["f1"].append(f1)

                res = {
                    "seed": seed,
                    "module": mod,
                    "accuracy_pct": round(acc, 2),
                    "f1_score": round(f1, 4),
                    "peak_vram_mb": eval_res.get("peak_vram_mb", 0.0),
                    "elapsed_s": round(time.time() - t0, 2),
                }

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                LOG.exception("Error on seed %d mod %s: %s", seed, mod, e)
                res = {"seed": seed, "module": mod, "error": str(e)}

            res_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Record detailed trial rows as expected by user prompt: module_name, seed, accuracy_pct, f1_score, peak_vram_mb
    trial_rows = []
    for seed in args.seeds:
        for mod in args.modules:
            fpath = out_dir / str(seed) / f"{mod}.json"
            if fpath.exists():
                try:
                    d = json.loads(fpath.read_text(encoding="utf-8"))
                    trial_rows.append({
                        "module_name": mod,
                        "seed": seed,
                        "accuracy_pct": d.get("accuracy_pct", 0.0),
                        "f1_score": d.get("f1_score", 0.0),
                        "peak_vram_mb": d.get("peak_vram_mb", 0.0),
                    })
                except Exception:
                    pass

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module_name", "seed", "accuracy_pct", "f1_score", "peak_vram_mb"])
        writer.writeheader()
        for r in trial_rows:
            writer.writerow(r)

    # Compute aggregate mean and standard deviation
    summary_rows = []
    for mod, data in module_runs.items():
        accs = data["acc"]
        f1s = data["f1"]
        mean_acc = round(float(np.mean(accs)), 2) if accs else 0.0
        std_acc = round(float(np.std(accs)), 2) if accs else 0.0
        mean_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
        std_f1 = round(float(np.std(f1s)), 4) if f1s else 0.0

        summary_rows.append({
            "module": mod,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "num_seeds": len(accs),
        })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "mean_acc", "std_acc", "mean_f1", "std_f1", "num_seeds"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"\n[DONE] Multi-seed evaluation completed. Results written to {summary_csv}")


if __name__ == "__main__":
    main()
