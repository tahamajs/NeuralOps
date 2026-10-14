#!/usr/bin/env python3
"""
Pareto Sweep: 72 Compressors on GPT-2 with WikiText-2 Perplexity.
Ensures fallback_to_original=False so bad delta/high error compression is faithfully exposed.
Captures:
  - Perplexity (PPL)
  - Compression Ratio / Param Reduction (%)
  - Peak VRAM (MB)
  - Latency (ms/token)
  - Tokens / sec throughput

Outputs:
  - JSON: results/v13_deep_empirical_suite/wikitext2_sweep/wikitext2_pareto_results.json
  - CSV: results/v13_deep_empirical_suite/wikitext2_sweep/wikitext2_pareto_results.csv
  - Markdown: results/v13_deep_empirical_suite/wikitext2_sweep/wikitext2_pareto_results.md
"""

import os
import sys
import time
import json
import csv
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.evaluation.eval_perplexity import compute_perplexity, build_calibration_loader
from neuralops.eval_all_models_datasets import apply_module_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WikiText2ParetoSweep")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
target_dtype = torch.float16 if device_str == "cuda" else torch.float32

out_dir = Path("results/v13_deep_empirical_suite/wikitext2_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Tokenizer & Base Calibration / Test Data once
logger.info("Initializing Tokenizer and WikiText-2 datasets...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    ds_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
except Exception:
    ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

calib_loader = build_calibration_loader(ds_train, tokenizer, num_samples=300, batch_size=4, max_length=128)
first_batch = calib_loader[0]["input_ids"].to(device_str)

# 2. Benchmark Baseline GPT-2
logger.info("Evaluating Baseline GPT-2...")
base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=target_dtype).to(device_str)
base_params = sum(p.numel() for p in base_model.parameters())

base_ppl, base_metrics = compute_perplexity(
    base_model, tokenizer, ds_test, seq_len=1024, stride=512, max_eval_tokens=2048, device=device_str
)
del base_model
torch.cuda.empty_cache()

logger.info("Baseline GPT-2 WikiText-2 PPL: %.2f | VRAM: %.1f MB", base_ppl, base_metrics["peak_vram_mb"])

records = [{
    "compressor": "baseline_gpt2",
    "status": "success",
    "perplexity": round(base_ppl, 2),
    "param_reduction_pct": 0.0,
    "compression_ratio": 1.0,
    "total_params": base_params,
    "peak_vram_mb": round(base_metrics["peak_vram_mb"], 1),
    "latency_ms_per_token": round(base_metrics["latency_ms_per_token"], 3),
    "throughput_tok_sec": round(base_metrics["throughput_tok_per_sec"], 1),
    "error": None,
}]

# Checkpoint resumption support
json_out = out_dir / "wikitext2_pareto_results.json"
completed_compressors = set()
if json_out.exists():
    try:
        with open(json_out, "r", encoding="utf-8") as f:
            saved_records = json.load(f)
        if isinstance(saved_records, list) and len(saved_records) > 0:
            records = saved_records
            completed_compressors = {r["compressor"] for r in records if r.get("status") in ("success", "failed")}
            logger.info("Resuming from checkpoint: found %d existing records (%d unique).", len(records), len(completed_compressors))
    except Exception as e:
        logger.warning("Could not read existing checkpoint: %s", e)

compressor_keys = sorted(list(COMPRESSOR_REGISTRY.keys()))
logger.info("Found %d registered compressors to evaluate.", len(compressor_keys))

for idx, comp_name in enumerate(compressor_keys, 1):
    if comp_name in completed_compressors:
        logger.info("[%d/%d] Skipping already completed compressor: %s", idx, len(compressor_keys), comp_name)
        continue

    logger.info("[%d/%d] Evaluating Compressor: %s ...", idx, len(compressor_keys), comp_name)
    torch.cuda.empty_cache()
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=target_dtype).to(device_str)
        
        # Enforce fallback_to_original=False
        model, stats = apply_module_to_model(
            model,
            comp_name,
            module_type="compressor",
            rank=32,
            device=torch.device(device_str),
            calibration_data=first_batch,
            calibration_loader=calib_loader,
            tokenizer=tokenizer,
            fallback_to_original=False,
        )

        curr_params = sum(p.numel() for p in model.parameters())
        comp_ratio = round(base_params / max(1, curr_params), 4)
        reduc_pct = round(100.0 * (1.0 - curr_params / max(1, base_params)), 2)

        ppl, metrics = compute_perplexity(
            model, tokenizer, ds_test, seq_len=1024, stride=512, max_eval_tokens=2048, device=device_str
        )

        del model
        torch.cuda.empty_cache()

        logger.info("  -> %s: PPL=%.2f | Reduc=%.1f%% | VRAM=%.1f MB", comp_name, ppl, reduc_pct, metrics["peak_vram_mb"])
        records.append({
            "compressor": comp_name,
            "status": "success",
            "perplexity": round(ppl, 2),
            "param_reduction_pct": reduc_pct,
            "compression_ratio": comp_ratio,
            "total_params": curr_params,
            "peak_vram_mb": round(metrics["peak_vram_mb"], 1),
            "latency_ms_per_token": round(metrics["latency_ms_per_token"], 3),
            "throughput_tok_sec": round(metrics["throughput_tok_per_sec"], 1),
            "error": None,
        })
    except Exception as e:
        logger.warning("  -> %s FAILED: %s", comp_name, e)
        records.append({
            "compressor": comp_name,
            "status": "failed",
            "perplexity": None,
            "param_reduction_pct": None,
            "compression_ratio": None,
            "total_params": None,
            "peak_vram_mb": None,
            "latency_ms_per_token": None,
            "throughput_tok_sec": None,
            "error": str(e)[:120],
        })
        torch.cuda.empty_cache()

    # Incremental write
    with open(out_dir / "wikitext2_pareto_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

# Write JSON
with open(out_dir / "wikitext2_pareto_results.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

# Write CSV
csv_cols = ["compressor", "status", "perplexity", "param_reduction_pct", "compression_ratio", "total_params", "peak_vram_mb", "latency_ms_per_token", "throughput_tok_sec", "error"]
with open(out_dir / "wikitext2_pareto_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    writer.writerows(records)

# Write Markdown
md_content = """# 72-Compressor WikiText-2 Pareto Frontier Sweep
**Hardware**: NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Configuration**: GPT-2 (124M) on WikiText-2 (`seq_len=1024`, `stride=512`, `rank=32`, `fallback_to_original=False`).

| Compressor | Status | Perplexity (PPL ↓) | Param Reduction (%) | Compression Ratio | Peak VRAM (MB) | Latency (ms/tok) | Throughput (tok/s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
for r in records:
    if r["status"] == "success":
        md_content += f"| **{r['compressor']}** | {r['status']} | {r['perplexity']} | {r['param_reduction_pct']}% | {r['compression_ratio']}x | {r['peak_vram_mb']} | {r['latency_ms_per_token']} | {r['throughput_tok_sec']} |\n"
    else:
        md_content += f"| **{r['compressor']}** | {r['status']} | - | - | - | - | - | - |\n"

with open(out_dir / "wikitext2_pareto_results.md", "w", encoding="utf-8") as f:
    f.write(md_content)

logger.info("WikiText-2 Pareto Sweep complete! Saved to %s", out_dir)
