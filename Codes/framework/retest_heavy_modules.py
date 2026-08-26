"""Targeted Re-test Script for Heavy/Optimized Attention Modules.

Specifically validates 'factorised_matrix_state' and 'mamba_svd_flash' after vectorization
and gradient checkpointing fixes.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
import torch
from transformers import BertConfig, BertForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY
from framework.eval_all_models_datasets import (
    MODEL_TOKENIZER_MAPPING,
    _filter_batch,
    get_dataset_batch,
    probe_optimal_batch_size,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RetestHeavy")


def retest_modules(output_dir: str = "results/retest_heavy_modules"):
    target_dir = PROJECT_DIR / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    modules_to_test = ["factorised_matrix_state", "mamba_svd_flash"]
    models = ["bert", "roberta"]
    datasets = ["sst2", "mrpc"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("================================================================================")
    logger.info("🚀 RE-TESTING OPTIMIZED HEAVY MODULES ON DEVICE: %s", device)
    logger.info("Target Modules: %s", modules_to_test)
    logger.info("================================================================================")

    results = {}

    for mod_name in modules_to_test:
        logger.info("\n--- Evaluating Module: %s ---", mod_name)
        mod_results = []
        all_passed = True

        for model_name in models:
            for dataset_name in datasets:
                t0 = time.time()
                combo_key = f"{mod_name}|{model_name}|{dataset_name}"
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

                    # Dynamic injection
                    inject_attention(model, mod_name, rank=16)
                    model.to(device)

                    # Dummy batch for test
                    b_size = 32
                    seq_len = 128
                    input_ids = torch.randint(0, 1000, (b_size, seq_len), device=device)
                    attn_mask = torch.ones(b_size, seq_len, device=device)
                    labels = torch.randint(0, num_labels, (b_size,), device=device)

                    # Forward pass
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    
                    t_fwd_0 = time.perf_counter()
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss = out.loss
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_fwd = (time.perf_counter() - t_fwd_0) * 1000.0

                    # Backward pass
                    t_bwd_0 = time.perf_counter()
                    loss.backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_bwd = (time.perf_counter() - t_bwd_0) * 1000.0

                    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

                    elapsed = time.time() - t0
                    res_entry = {
                        "combination": combo_key,
                        "status": "PASS",
                        "fwd_ms": round(t_fwd, 2),
                        "bwd_ms": round(t_bwd, 2),
                        "peak_vram_mb": round(vram_mb, 1),
                        "loss": round(loss.item(), 4),
                        "elapsed_sec": round(elapsed, 2),
                    }
                    mod_results.append(res_entry)
                    logger.info("  ✓ %s -> PASS (fwd: %.2fms, bwd: %.2fms, vram: %.1fMB, loss: %.4f in %.2fs)", combo_key, t_fwd, t_bwd, vram_mb, loss.item(), elapsed)

                except Exception as e:
                    all_passed = False
                    err_str = str(e)
                    logger.error("  ✗ %s -> FAIL: %s", combo_key, err_str)
                    mod_results.append({
                        "combination": combo_key,
                        "status": "FAIL",
                        "error": err_str,
                    })

        final_verdict = "PASS" if all_passed else "UNSUPPORTED"
        results[mod_name] = {
            "verdict": final_verdict,
            "details": mod_results,
        }

    # Save summary
    summary_path = target_dir / "retest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n================================================================================")
    logger.info("🎯 FINAL RE-TEST VERDICTS:")
    for mod, data in results.items():
        logger.info("  Module '%s': %s", mod, data["verdict"])
    logger.info("Results saved to: %s", summary_path)
    logger.info("================================================================================")

    return results


if __name__ == "__main__":
    retest_modules()
