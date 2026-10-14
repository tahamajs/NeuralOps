#!/usr/bin/env python3
"""
Targeted Extended Fine-Tuning Sweep for Low-Accuracy / Chance-Level Modules on RoBERTa-base (ssh gpu).
Trains each module for 3 epochs with 400 samples on SST-2 to ensure proper gradient convergence.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobertaExtendedSweep")

# Modules that scored around chance level (~0.48 - 0.50) on single-epoch/tiny-sample RoBERTa run
LOW_ACC_ROBERTA_MODULES = [
    "dynamic_conv_attention",
    "hamming_attention",
    "sink_attention",
    "gated_delta_net",
    "gated_deltanet_block",
    "granite_hybrid",
    "hgrn2_gated_deltanet",
    "hgrn2_gated_outer",
    "hgrn2_outer",
    "mamba_block",
    "mamba_svd_flash",
    "selective_ssm",
    "simple_ssm",
    "svd_mamba",
    "ttt_linear_attention",
]

def main():
    parser = argparse.ArgumentParser(description="Run extended fine-tuning for RoBERTa modules.")
    parser.add_argument("--modules", nargs="*", default=LOW_ACC_ROBERTA_MODULES)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--max-eval-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/roberta_extended_convergence")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting extended RoBERTa fine-tuning for %d modules on %s...", len(args.modules), args.device)
    python_bin = sys.executable
    results = []

    for idx, mod in enumerate(args.modules, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Extended Training: %s (epochs=%d, samples=%d)", idx, len(args.modules), mod, args.epochs, args.max_samples)
        logger.info("=" * 60)

        cmd = [
            python_bin, "-m", "neuralops.benchmarks", "run",
            "--module", mod,
            "--model", "roberta-base",
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

    logger.info("Finished RoBERTa extended training: %d/%d succeeded.", sum(1 for r in results if r["status"] == "success"), len(results))

if __name__ == "__main__":
    main()
