#!/usr/bin/env python3
"""
RoBERTa-Base 72-Compressor Fine-Tuning Sweep on SST-2.
Evaluates all 72 compressors on RoBERTa-base with SST-2 classification.
Captures:
  - Accuracy & F1
  - Evaluation Loss
  - Param Reduction (%)
  - Peak VRAM (MB)
  - Duration (s)

Outputs:
  - results/v13_deep_empirical_suite/roberta_72_compressors/results.json
  - results/v13_deep_empirical_suite/roberta_72_compressors/results.csv
  - results/v13_deep_empirical_suite/roberta_72_compressors/results.md
"""

import os
import sys
import time
import json
import csv
import logging
from pathlib import Path

import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.benchmarks.core.data_loader import load_dataset_by_name, make_dataloader
from neuralops.benchmarks.core.training import finetune_classification
from neuralops.benchmarks.core.evaluation import evaluate_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Roberta72CompressorSweep")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("results/v13_deep_empirical_suite/roberta_72_compressors")
out_dir.mkdir(parents=True, exist_ok=True)

logger.info("Initializing Tokenizer and Datasets for roberta-base / sst2...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
train_ds = load_dataset_by_name("sst2", tokenizer=tokenizer, max_length=64, split="train", max_samples=300)
eval_ds = load_dataset_by_name("sst2", tokenizer=tokenizer, max_length=64, split="validation", max_samples=100)

train_loader = make_dataloader(train_ds, batch_size=8, shuffle=True)
eval_loader = make_dataloader(eval_ds, batch_size=16, shuffle=False)
calib_batch = next(iter(train_loader))[0].to(device_str)

# Evaluate Baseline
logger.info("Evaluating Baseline RoBERTa-base...")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device_str)
base_params = sum(p.numel() for p in base_model.parameters())
torch.cuda.reset_peak_memory_stats()
train_metrics = finetune_classification(base_model, train_loader, epochs=1, lr=2e-5, device=device_str)
eval_metrics = evaluate_classification(base_model, eval_loader, device=device_str)
base_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device_str == "cuda" else 0.0

records = [{
    "compressor": "baseline_roberta",
    "status": "success",
    "accuracy": round(eval_metrics.get("accuracy", 0.0), 4),
    "f1": round(eval_metrics.get("f1", 0.0), 4),
    "loss": round(eval_metrics.get("loss", 0.0), 4),
    "param_reduction_pct": 0.0,
    "compression_ratio": 1.0,
    "total_params": base_params,
    "peak_vram_mb": round(base_peak_vram, 1),
    "error": None,
}]
del base_model
torch.cuda.empty_cache()

compressor_keys = sorted(list(COMPRESSOR_REGISTRY.keys()))
logger.info("Found %d registered compressors to evaluate on RoBERTa-base.", len(compressor_keys))

for idx, comp_name in enumerate(compressor_keys, 1):
    logger.info("[%d/%d] Evaluating Compressor: %s on RoBERTa-base ...", idx, len(compressor_keys), comp_name)
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device_str)
        model, stats = apply_module_to_model(
            model,
            comp_name,
            module_type="compressor",
            rank=32,
            device=torch.device(device_str),
            calibration_data=calib_batch,
            tokenizer=tokenizer,
            fallback_to_original=False,
        )
        curr_params = sum(p.numel() for p in model.parameters())
        comp_ratio = round(base_params / max(1, curr_params), 4)
        reduc_pct = round(100.0 * (1.0 - curr_params / max(1, base_params)), 2)

        torch.cuda.reset_peak_memory_stats()
        train_m = finetune_classification(model, train_loader, epochs=1, lr=2e-5, device=device_str)
        eval_m = evaluate_classification(model, eval_loader, device=device_str)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device_str == "cuda" else 0.0
        dur = round(time.time() - t0, 2)

        records.append({
            "compressor": comp_name,
            "status": "success",
            "accuracy": round(eval_m.get("accuracy", 0.0), 4),
            "f1": round(eval_m.get("f1", 0.0), 4),
            "loss": round(eval_m.get("loss", 0.0), 4),
            "param_reduction_pct": reduc_pct,
            "compression_ratio": comp_ratio,
            "total_params": curr_params,
            "peak_vram_mb": round(peak_vram, 1),
            "duration_s": dur,
            "error": None,
        })
        logger.info("  -> %s SUCCESS: acc=%.4f, f1=%.4f, vram=%.1f MB, time=%.1fs", comp_name, eval_m.get("accuracy", 0.0), eval_m.get("f1", 0.0), peak_vram, dur)
        del model
    except Exception as e:
        logger.warning("  -> %s FAILED: %s", comp_name, e)
        dur = round(time.time() - t0, 2)
        records.append({
            "compressor": comp_name,
            "status": "failed",
            "accuracy": None,
            "f1": None,
            "loss": None,
            "param_reduction_pct": None,
            "compression_ratio": None,
            "total_params": None,
            "peak_vram_mb": None,
            "duration_s": dur,
            "error": str(e)[:120],
        })
    torch.cuda.empty_cache()

    # Incremental write
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

# Final outputs
csv_cols = ["compressor", "status", "accuracy", "f1", "loss", "param_reduction_pct", "compression_ratio", "total_params", "peak_vram_mb", "duration_s", "error"]
with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    writer.writerows(records)

md_content = """# 72-Compressor RoBERTa-Base SST-2 Fine-Tuning Sweep
**Hardware**: NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Configuration**: RoBERTa-base on SST-2 (`rank=32`, `fallback_to_original=False`).

| Compressor | Status | Accuracy (↑) | F1 Score (↑) | Loss (↓) | Param Reduction (%) | Compression Ratio | Peak VRAM (MB) | Duration (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
for r in records:
    if r["status"] == "success":
        md_content += f"| **{r['compressor']}** | {r['status']} | {r['accuracy']} | {r['f1']} | {r['loss']} | {r['param_reduction_pct']}% | {r['compression_ratio']}x | {r['peak_vram_mb']} | {r['duration_s']}s |\n"
    else:
        md_content += f"| **{r['compressor']}** | {r['status']} | - | - | - | - | - | - | {r['duration_s']}s |\n"

with open(out_dir / "results.md", "w", encoding="utf-8") as f:
    f.write(md_content)

logger.info("RoBERTa 72-Compressor sweep complete! Saved to %s", out_dir)
