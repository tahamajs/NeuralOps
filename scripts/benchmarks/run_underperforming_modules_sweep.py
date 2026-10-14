#!/usr/bin/env python3
"""
Targeted Re-training & Debugging Sweep for Underperforming Modules on tahoora (GTX 1060).
Executes full fine-tuning (e.g. 3 epochs, batch_size=4, seq_len=64, 500 samples)
to give underperforming modules enough optimization steps to converge.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of modules that had low accuracy (<= 0.45) or high loss (> 2.0) during single-epoch/tiny-sample sweep
UNDERPERFORMING_MODULES = [
    # Top critical low-accuracy modules
    "kda_attention",
    "block_transformer_hiba",
    "aegis_hybrid",
    "tri_branch_hybrid",
    "fusion_attn",
    "phantom",
    "aegis_v5",
    "split_head_hybrid",
    "padre_attention",
    "qcqa_attention",
    "triton_cosformer",
    "zeros_attention",
    "hgrn2_gated_outer",
    "hgrn2_outer",
    "optimized_hybrid_attention",
    "chela_slay",
    "rpc_attention",
    "griffin_ssd",
    "enlighten_compressor",
    "grecodec_compressor",
    "mgie_budget",
    "entropy_budget_allocation",
    "frandi_compressor",
    "astro_quantizer",
    "rpca_compressor",
    "drone_compressor",
    "fwsvd_compressor",
]

def main():
    parser = argparse.ArgumentParser(description="Run full fine-tuning for underperforming modules.")
    parser.add_argument("--modules", nargs="*", default=UNDERPERFORMING_MODULES, help="List of modules to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=500, help="Max training samples")
    parser.add_argument("--max-eval-samples", type=int, default=100, help="Max eval samples")
    parser.add_argument("--output-dir", type=str, default="results/full_training_underperforming", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting targeted full fine-tuning sweep for %d modules on %s...", len(args.modules), args.device)
    
    python_bin = sys.executable
    results = []

    for idx, mod in enumerate(args.modules, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Training module: %s (epochs=%d, samples=%d)", idx, len(args.modules), mod, args.epochs, args.max_samples)
        logger.info("=" * 60)

        cmd = [
            python_bin, "-m", "neuralops.benchmarks", "run",
            "--module", mod,
            "--model", "gpt2",
            "--dataset", "sst2",
            "--max-samples", str(args.max_samples),
            "--max-eval-samples", str(args.max_eval_samples),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seq-len", "64",
            "--device", args.device,
            "--output-dir", str(out_dir),
            "--no-deduplicate",
            "--force-rerun",
        ]

        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            logger.info("Module %s training completed successfully.", mod)
            results.append({"module": mod, "status": "success"})
        except subprocess.CalledProcessError as e:
            logger.error("Module %s failed with exit code %d:\n%s", mod, e.returncode, e.stdout[-500:] if e.stdout else "")
            results.append({"module": mod, "status": "failed", "error": str(e)})

    logger.info("Finished targeted sweep: %d/%d succeeded.", sum(1 for r in results if r["status"] == "success"), len(results))

if __name__ == "__main__":
    main()
