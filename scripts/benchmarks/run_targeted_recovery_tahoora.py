#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run targeted recovery tahoora module.
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_MODULES = [
    "block_transformer_hiba",
    "kda_attention",
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
    "rpca_compressor",
    "enlighten_compressor",
    "grecodec_compressor",
    "mgie_budget",
    "entropy_budget_allocation",
    "frandi_compressor",
    "astro_quantizer",
    "drone_compressor",
    "fwsvd_compressor",
    "huffman_coder",
]


def main():
    """Execute main routine."""
    python_bin = sys.executable
    out_dir = Path("results/full_training_underperforming")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting targeted full fine-tuning queue (%d modules)...", len(TARGET_MODULES))

    for idx, mod in enumerate(TARGET_MODULES, 1):
        logger.info("=" * 60)
        logger.info("===> [%d/%d] Launching full training for: %s", idx, len(TARGET_MODULES), mod)
        logger.info("=" * 60)
        cmd = [
            python_bin,
            "-m",
            "neuralops.benchmarks",
            "run",
            "--module",
            mod,
            "--model",
            "gpt2",
            "--dataset",
            "sst2",
            "--max-samples",
            "500",
            "--max-eval-samples",
            "100",
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--seq-len",
            "64",
            "--device",
            "cuda",
            "--output-dir",
            str(out_dir),
            "--no-deduplicate",
            "--force-rerun",
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            lines = [l for l in res.stdout.splitlines() if "Metrics:" in l or "[SUCCESS]" in l or "[FAIL]" in l]
            for l in lines:
                logger.info("  %s", l)
        except subprocess.CalledProcessError as e:
            logger.error("  FAILED: %s (exit %d)", mod, e.returncode)
            tail = "\n".join(e.stdout.splitlines()[-10:])
            logger.error("  Error log:\n%s", tail)


if __name__ == "__main__":
    main()
