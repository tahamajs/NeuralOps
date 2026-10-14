#!/usr/bin/env python3
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Underperforming modules on RoBERTa-base (accuracy <= 0.516)
TARGET_MODULES_GPU = [
    # Lowest accuracy & SSM modules
    "sink_attention",
    "hamming_attention",
    "dynamic_conv_attention",
    "selective_ssm",
    "simple_ssm",
    "granite_hybrid",
    "svd_mamba",
    "hgrn2_gated_deltanet",
    "mamba_svd_flash",
    "ttt_linear_attention",
    "gated_delta_net",
    "gated_deltanet_block",
    "upi_stuffed_mamba",
    "mamba_block",
    "hgrn2_gated_outer",
    "hgrn2_outer",
    "galerkin_attention",
    "jamba_info",
    "positive_kernel_linear_attention",
    "svd_linear",
    "halo_regla",
    "hybrid_attention",
    "minimal_nova",
    "hgrn2_gated_layer",
    "svd_llm_v2_attention",
    "compressed_attention",
    "spectron",
    "svd_seq",
    "compressed_mha_attention",
    "expert_choice_moe",
]

def main():
    python_bin = sys.executable
    out_dir = Path("results/full_training_underperforming_roberta")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting targeted full fine-tuning queue on GPU RTX 4070 (%d modules)...", len(TARGET_MODULES_GPU))
    
    for idx, mod in enumerate(TARGET_MODULES_GPU, 1):
        logger.info("=" * 60)
        logger.info("===> [%d/%d] Launching RoBERTa-base full training for: %s", idx, len(TARGET_MODULES_GPU), mod)
        logger.info("=" * 60)
        cmd = [
            python_bin, "-m", "neuralops.benchmarks", "run",
            "--module", mod,
            "--model", "roberta-base",
            "--dataset", "sst2",
            "--max-samples", "500",
            "--max-eval-samples", "100",
            "--epochs", "3",
            "--batch-size", "8",
            "--seq-len", "64",
            "--device", "cuda",
            "--output-dir", str(out_dir),
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
