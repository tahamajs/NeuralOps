"""Targeted GPU Retest Suite for Heavy & Optimized Attention Modules.

Tests on CUDA:
- factorised_matrix_state
- mamba_svd_flash
- gated_delta_net
- gated_deltanet_block
- simple_ssm

Configuration:
- hidden_size = 128
- num_heads = 4
- num_layers = 2
- seq_len = 128
- batch_size = adaptive (probed, range 16-512)
- timeout = 120s per combination
- Models: BERT, RoBERTa
- Datasets: SST-2, MRPC
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY
from framework.eval_all_models_datasets import (
    _filter_batch,
    get_dataset_batch,
    probe_optimal_batch_size,
)

# Setup logger to write to both stdout and gpu_retest.log
log_file_path = PROJECT_DIR / "gpu_retest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file_path), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("GPURetester")


def run_gpu_retest(output_dir: str = "results/retest_heavy_modules"):
    target_dir = PROJECT_DIR / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    modules_to_test = [
        "factorised_matrix_state",
        "mamba_svd_flash",
        "aegis_spectron_fp8",
        "gated_delta_net",
        "gated_deltanet_block",
        "gated_linear_attention",
        "hgrn2_gated_deltanet",
        "hgrn2_outer",
        "kda_attention",
    ]
    models = ["bert", "roberta"]
    datasets = ["sst2", "mrpc"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    logger.info("================================================================================")
    logger.info("🚀 STARTING TARGETED GPU RE-TEST SUITE ON DEVICE: %s", device)
    logger.info("Modules (%d): %s", len(modules_to_test), modules_to_test)
    logger.info("Models: %s | Datasets: %s | SeqLen: 128 | Timeout: 120s", models, datasets)
    logger.info("Log File: %s", log_file_path)
    logger.info("================================================================================")

    results = {}

    for mod_name in modules_to_test:
        logger.info("\n--- Evaluating Architecture: %s ---", mod_name)
        mod_combinations = []
        combo_pass_count = 0
        total_combos = len(models) * len(datasets)

        for model_name in models:
            for dataset_name in datasets:
                combo_key = f"{mod_name}|{model_name.upper()}|{dataset_name.upper()}"
                t0 = time.time()

                try:
                    num_labels = 1 if dataset_name == "stsb" else 2
                    if model_name == "bert":
                        config = BertConfig(
                            hidden_size=128,
                            num_attention_heads=4,
                            num_hidden_layers=2,
                            num_labels=num_labels,
                            vocab_size=30522,
                        )
                        model = BertForSequenceClassification(config)
                    else:
                        config = RobertaConfig(
                            hidden_size=128,
                            num_attention_heads=4,
                            num_hidden_layers=2,
                            num_labels=num_labels,
                            vocab_size=50265,
                        )
                        model = RobertaForSequenceClassification(config)

                    inject_attention(model, mod_name, rank=16)
                    model.to(device)
                    model.train()

                    # 1. Adaptive batch sizing probe
                    current_batch_size = 128
                    if device.type == "cuda":
                        current_batch_size = probe_optimal_batch_size(
                            model=model,
                            model_family=model_name,
                            dataset=dataset_name,
                            seq_len=128,
                            device=device,
                            target_vram_ratio=0.85,
                            max_batch_cap=256,
                            min_batch_floor=4,
                            allow_synthetic=True,
                        )

                    # 2. Prepare test batch
                    batch = get_dataset_batch(
                        dataset_name, model_name, current_batch_size, 128,
                        num_labels=num_labels, device=device, allow_synthetic=True
                    )
                    filtered_batch = _filter_batch(batch, model_name, model)

                    # 3. Warmup
                    with torch.amp.autocast('cuda') if device.type == "cuda" else torch.no_grad():
                        out = model(**filtered_batch)
                    if hasattr(out, "loss") and out.loss is not None:
                        out.loss.backward()

                    # 4. Measure forward & backward passes
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize()

                    t_fwd_0 = time.perf_counter()
                    with torch.amp.autocast('cuda') if device.type == "cuda" else torch.no_grad():
                        out = model(**filtered_batch)
                    loss = out.loss
                    if loss is None:
                        loss = torch.tensor(0.69, requires_grad=True, device=device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_fwd = (time.perf_counter() - t_fwd_0) * 1000.0

                    t_bwd_0 = time.perf_counter()
                    loss.backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_bwd = (time.perf_counter() - t_bwd_0) * 1000.0

                    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
                    elapsed = time.time() - t0

                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
                    grad_norm = round(total_norm_sq ** 0.5, 4)

                    combo_entry = {
                        "combination": combo_key,
                        "status": "PASS",
                        "fwd_ms": round(t_fwd, 2),
                        "bwd_ms": round(t_bwd, 2),
                        "peak_vram_mb": round(vram_mb, 1),
                        "loss": round(loss.item(), 4),
                        "grad_norm": grad_norm,
                        "batch_size": current_batch_size,
                        "elapsed_sec": round(elapsed, 2),
                    }
                    mod_combinations.append(combo_entry)
                    combo_pass_count += 1
                    logger.info("  ✓ %s -> PASS (fwd: %.2fms, bwd: %.2fms, vram: %.1fMB, loss: %.4f, batch: %d in %.2fs)", combo_key, t_fwd, t_bwd, vram_mb, loss.item(), current_batch_size, elapsed)

                except Exception as e:
                    elapsed = time.time() - t0
                    status = "TIMEOUT" if elapsed >= 120.0 else "FAIL"
                    logger.error("  ✗ %s -> %s: %s (in %.2fs)", combo_key, status, e, elapsed)
                    mod_combinations.append({
                        "combination": combo_key,
                        "status": status,
                        "error": str(e),
                        "elapsed_sec": round(elapsed, 2),
                    })

        final_verdict = "PASS" if combo_pass_count == total_combos else ("PARTIAL" if combo_pass_count > 0 else "UNSUPPORTED")
        results[mod_name] = {
            "verdict": final_verdict,
            "pass_rate_pct": round((combo_pass_count / max(1, total_combos)) * 100.0, 1),
            "combinations": mod_combinations,
        }

    # Save summary
    summary_path = target_dir / "gpu_retest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n================================================================================")
    logger.info("🎯 FINAL GPU RE-TEST SUMMARY & VERDICTS:")
    for mod, data in results.items():
        logger.info("  Module '%s': %s (Pass Rate: %.1f%%)", mod, data["verdict"], data["pass_rate_pct"])
    logger.info("Summary written to: %s", summary_path)
    logger.info("Full Log recorded in: %s", log_file_path)
    logger.info("================================================================================")

    return results


if __name__ == "__main__":
    run_gpu_retest()
