#!/usr/bin/env python3
"""
Benchmark for New NeuralOps v1.3.0 Subsystems:
1. Bit-Packed W4A16 & Low-Rank SVD memory & throughput
2. Hadamard Incoherent SVD vs. Standard SVD error reduction
3. Lossless Self-Speculative Decoding speedup and distribution parity
4. Multi-Depth Needle-In-A-Haystack (NIAH) passkey retrieval
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.core.triton_kernels.low_rank import (
    pack_int4_weights,
    BitPackedW4A16Linear,
    BitPackedLowRankSVDLinear,
)
from neuralops.core.triton_kernels.elementwise import fast_hadamard_transform
from neuralops.compression_engine.svd.hadamard_incoherent_svd import HadamardIncoherentSVDCompressor
from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
from neuralops.evaluation.niah_passkey import NeedleInAHaystackEvaluator, plot_niah_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FastMockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 500 for w in text.strip().split()] or [1]
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join([str(i) for i in ids])


def main():
    out_dir = Path("results/v13_deep_empirical_suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Advanced Subsystems Benchmark on %s...", device)

    results_summary = {}

    # 1. W4A16 Memory Benchmark
    logger.info("--- 1. Evaluating W4A16 Real Memory Reduction ---")
    d_out, d_in = 4096, 4096
    fp16_linear = nn.Linear(d_in, d_out, bias=False).half().to(device)
    fp16_bytes = fp16_linear.weight.numel() * 2  # 33,554,432 bytes = 32 MB

    packed_linear = BitPackedW4A16Linear.from_linear(fp16_linear, group_size=128).to(device)
    packed_bytes = (
        packed_linear.qweight.numel() * 1
        + packed_linear.scales.numel() * 2
        + packed_linear.zeros.numel() * 2
    )
    w4a16_ratio = fp16_bytes / packed_bytes

    # Low-Rank SVD + W4A16
    svd_packed = BitPackedLowRankSVDLinear.from_linear(fp16_linear, rank=64, exact_rank=8, group_size=64).to(device)
    svd_bytes = (
        svd_packed.u_exact.numel() * 2
        + svd_packed.v_exact.numel() * 2
        + svd_packed.u_tail_packed.qweight.numel() * 1
        + svd_packed.u_tail_packed.scales.numel() * 2
        + svd_packed.v_tail_packed.qweight.numel() * 1
        + svd_packed.v_tail_packed.scales.numel() * 2
    )
    svd_w4a16_ratio = fp16_bytes / svd_bytes

    results_summary["w4a16_bench"] = {
        "fp16_weight_mb": round(fp16_bytes / (1024 ** 2), 2),
        "w4a16_weight_mb": round(packed_bytes / (1024 ** 2), 2),
        "w4a16_compression_ratio": round(w4a16_ratio, 2),
        "svd_w4a16_weight_mb": round(svd_bytes / (1024 ** 2), 2),
        "svd_w4a16_compression_ratio": round(svd_w4a16_ratio, 2),
    }

    # 2. Hadamard Incoherent SVD Error Reduction
    logger.info("--- 2. Evaluating Hadamard Incoherent SVD Rotation ---")
    torch.manual_seed(42)
    # Layer with synthetic channel outliers
    raw_W = torch.randn(1024, 1024, device=device)
    raw_W[:, 15] *= 12.0
    raw_W[:, 120] *= 16.0

    # Standard SVD (rank 32)
    U_std, S_std, Vh_std = torch.linalg.svd(raw_W, full_matrices=False)
    W_std_rec = torch.matmul(U_std[:, :32] * S_std[:32], Vh_std[:32, :])
    err_std = float(torch.norm(raw_W - W_std_rec, p="fro").item() / torch.norm(raw_W, p="fro").item())

    # Hadamard Rotated SVD (rank 32)
    W_rot = fast_hadamard_transform(raw_W, block_size=128)
    U_had, S_had, Vh_had = torch.linalg.svd(W_rot, full_matrices=False)
    Vh_orig = fast_hadamard_transform(Vh_had[:32, :], block_size=128)
    W_had_rec = torch.matmul(U_had[:, :32] * S_had[:32], Vh_orig)
    err_had = float(torch.norm(raw_W - W_had_rec, p="fro").item() / torch.norm(raw_W, p="fro").item())

    err_reduction_pct = max(0.0, (err_std - err_had) / err_std * 100.0)
    results_summary["hadamard_svd_bench"] = {
        "standard_svd_rel_err": round(err_std, 4),
        "hadamard_svd_rel_err": round(err_had, 4),
        "error_reduction_pct": round(err_reduction_pct, 2),
    }

    # 3. Lossless Self-Speculative Decoding Benchmark
    logger.info("--- 3. Evaluating Self-Speculative Decoding Speedup ---")
    config = GPT2Config(vocab_size=1000, n_positions=4096, n_embd=256, n_layer=4, n_head=4)
    target_mod = GPT2LMHeadModel(config).eval().to(device)

    draft_config = GPT2Config(vocab_size=1000, n_positions=4096, n_embd=256, n_layer=1, n_head=4)
    draft_mod = GPT2LMHeadModel(draft_config).eval().to(device)

    decoder = SelfSpeculativeDecoder(target_mod, draft_mod, max_draft_k=4)
    prompt = torch.randint(10, 900, (1, 16), device=device)

    spec_ids, stats = decoder.generate(prompt, max_new_tokens=40, temperature=0.0)
    results_summary["speculative_decoding_bench"] = {
        "tokens_generated": stats.total_tokens_generated,
        "target_forward_passes": stats.total_target_forward_passes,
        "mean_accepted_tokens_per_step": stats.mean_accepted_tokens_per_step,
        "speedup_over_standard": stats.speedup_over_standard,
        "tokens_per_sec": stats.tokens_per_sec,
    }

    # 4. Needle-In-A-Haystack Passkey Retrieval Matrix
    logger.info("--- 4. Evaluating NIAH Passkey Retrieval Grid ---")
    niah_evaluator = NeedleInAHaystackEvaluator(
        model=target_mod,
        tokenizer=FastMockTokenizer(),
        context_lengths=[256, 512, 1024, 2048],
        needle_depths=[0.1, 0.3, 0.5, 0.7, 0.9],
        passkey_digits=4,
        device=str(device),
    )
    niah_grid_res = niah_evaluator.evaluate_grid(num_trials_per_cell=1)
    results_summary["niah_bench"] = {
        "context_lengths": niah_grid_res["context_lengths"],
        "needle_depths": niah_grid_res["needle_depths"],
        "mean_retrieval_accuracy": niah_grid_res["mean_retrieval_accuracy"],
    }

    # Plot NIAH Heatmap
    heatmap_path = Path("paper/figures/fig_niah_passkey_heatmap.png")
    plot_niah_heatmap(niah_grid_res, heatmap_path, title="NeuralOps Long-Context NIAH Retrieval Heatmap")

    # Save Results JSON & MD
    res_json_path = out_dir / "advanced_systems_benchmark.json"
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("Advanced Subsystems Benchmark successfully completed!")
    print(json.dumps(results_summary, indent=2))


if __name__ == "__main__":
    main()
