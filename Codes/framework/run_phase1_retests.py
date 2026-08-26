"""Standalone Diagnostic & Profiling Suite for Heavy / Optimized Attention Modules.

Tests:
- factorised_matrix_state
- mamba_svd_flash
- gated_delta_net
- gated_deltanet_block
- simple_ssm (profiling)
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Phase1Retester")


def run_retests(output_dir: str = "results/retest_heavy_modules"):
    target_dir = PROJECT_DIR / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    modules_to_test = [
        "factorised_matrix_state",
        "mamba_svd_flash",
        "gated_delta_net",
        "gated_deltanet_block",
        "simple_ssm",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("================================================================================")
    logger.info("🔬 RUNNING STANDALONE DIAGNOSTIC RE-TESTS ON DEVICE: %s", device)
    logger.info("Target Modules (%d): %s", len(modules_to_test), modules_to_test)
    logger.info("================================================================================")

    results = {}

    for mod_name in modules_to_test:
        logger.info("\n--- Profiling Module: %s ---", mod_name)
        try:
            cfg = BertConfig(
                hidden_size=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                num_labels=2,
                vocab_size=30522,
            )
            model = BertForSequenceClassification(cfg)
            inject_attention(model, mod_name, rank=16)
            model.to(device)
            model.train()

            b_size = 2
            seq_len = 64
            input_ids = torch.randint(0, 1000, (b_size, seq_len), device=device)
            mask = torch.ones(b_size, seq_len, device=device)
            labels = torch.randint(0, 2, (b_size,), device=device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Warmup
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward()

            # Benchmark 2 steps
            fwd_times, bwd_times = [], []
            losses = []
            for step in range(2):
                model.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = out.loss
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fwd = (time.perf_counter() - t0) * 1000.0
                fwd_times.append(t_fwd)
                losses.append(loss.item())

                t1 = time.perf_counter()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_bwd = (time.perf_counter() - t1) * 1000.0
                bwd_times.append(t_bwd)

            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
            avg_fwd = round(float(sum(fwd_times) / len(fwd_times)), 2)
            avg_bwd = round(float(sum(bwd_times) / len(bwd_times)), 2)
            final_loss = round(float(losses[-1]), 4)

            # Check gradients
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += float(p.grad.data.norm(2).item() ** 2)
            total_grad_norm = round(total_grad_norm ** 0.5, 4)

            verdict = "PASS" if total_grad_norm > 0.0 and not torch.isnan(torch.tensor(final_loss)) else "FAIL"

            res_entry = {
                "module_name": mod_name,
                "status": verdict,
                "fwd_ms": avg_fwd,
                "bwd_ms": avg_bwd,
                "peak_vram_mb": round(vram_mb, 1),
                "final_loss": final_loss,
                "grad_norm": total_grad_norm,
            }
            results[mod_name] = res_entry
            logger.info("  ✓ %s -> %s (fwd: %.2fms, bwd: %.2fms, grad_norm: %.4f, loss: %.4f)", mod_name, verdict, avg_fwd, avg_bwd, total_grad_norm, final_loss)

        except Exception as e:
            logger.error("  ✗ %s -> FAIL: %s", mod_name, e)
            results[mod_name] = {
                "module_name": mod_name,
                "status": "FAIL",
                "error": str(e),
            }

    report_path = target_dir / "retest_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("================================================================================")
    logger.info("🎯 DIAGNOSTIC RE-TEST COMPLETED. Saved to: %s", report_path)
    logger.info("================================================================================")
    return results


if __name__ == "__main__":
    run_retests()
