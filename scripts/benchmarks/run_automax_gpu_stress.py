#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
High-Utilization GPU Saturation & Continuous AutoMax Stress Benchmark.

Saturates NVIDIA RTX 3090 (24GB VRAM, 350W TDP) with dense multi-head attention
and feed-forward Transformer blocks, continuous forward+backward gradient steps,
AMP Tensor Core execution, and real-time telemetry profiling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoMaxStress")


class SyntheticLMDataset(Dataset):
    """Infinite or large synthetic token sequence dataset for continuous throughput stress."""

    def __init__(self, num_samples: int = 10000, seq_len: int = 512, vocab_size: int = 32000) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate deterministic synthetic sequences without CPU overhead
        torch.manual_seed(idx % 1000)
        x = torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": x, "labels": y}


class HeavyTransformerLM(nn.Module):
    """Deep and wide Transformer language model to fully saturate Tensor Cores and HBM/VRAM."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1536,
        n_layers: int = 12,
        n_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.tok_emb(input_ids)
        h = self.layers(h)
        h = self.norm(h)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"logits": logits, "loss": loss}


def get_gpu_telemetry(device_idx: int = 0) -> dict[str, float]:
    """Retrieve physical device power, memory, and utilization via torch.cuda."""
    info: dict[str, float] = {}
    if not torch.cuda.is_available():
        return info

    total_mb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)
    alloc_mb = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
    res_mb = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
    max_alloc_mb = torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024)

    info["vram_total_gb"] = round(total_mb / 1024, 2)
    info["vram_alloc_gb"] = round(alloc_mb / 1024, 2)
    info["vram_reserved_gb"] = round(res_mb / 1024, 2)
    info["vram_max_alloc_gb"] = round(max_alloc_mb / 1024, 2)
    info["vram_util_pct"] = round((alloc_mb / total_mb) * 100, 1)

    # Attempt to query NVML for real-time power (Watts)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        info["power_watts"] = round(power_mw / 1000.0, 1)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        info["gpu_sm_util_pct"] = float(util.gpu)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        info["temp_c"] = float(temp)
    except Exception:
        pass

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoMax Continuous GPU Saturation Benchmark")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size per step")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=1536, help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=12, help="Number of Transformer layers")
    parser.add_argument("--steps", type=int, default=100, help="Continuous training steps")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (FP16/BF16)")
    parser.add_argument("--target-vram-pct", type=float, default=85.0, help="Target VRAM saturation percentage")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This benchmark strictly requires physical GPU hardware.")
        return 3

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 72)
    print("  [AutoMax] NeuralOps Continuous GPU Saturation Engine")
    print("=" * 72)
    print(f"  Target Accelerator : {device_name} ({total_vram_gb:.2f} GB VRAM)")
    print(f"  Model Architecture : {args.layers}L x {args.d_model}D (seq_len={args.seq_len})")
    print(f"  Batch Size         : {args.batch_size}")
    print(f"  Target Steps       : {args.steps}")
    print(f"  Target VRAM Goal   : {args.target_vram_pct:.1f}% (~{total_vram_gb * args.target_vram_pct / 100:.1f} GB)")
    print("=" * 72)

    # Enable Tensor Core matmul acceleration
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 1. Instantiate Model
    logger.info("Initializing HeavyTransformerLM on device...")
    model = HeavyTransformerLM(
        vocab_size=32000,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=16,
        d_ff=args.d_model * 4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    logger.info(f"Model parameters: {param_count:,} ({param_mb:.1f} MB FP32 weights)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 2. Build Dataset & Loader
    dataset = SyntheticLMDataset(num_samples=10000, seq_len=args.seq_len, vocab_size=32000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Prevent IPC deadlock
        pin_memory=True,
    )
    data_iter = iter(loader)

    # 3. Warmup Phase
    logger.info(f"Running {args.warmup_steps} warmup steps to populate CUDA kernels & buffers...")
    for _ in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(input_ids, labels=labels)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    logger.info("Warmup complete. Commencing continuous high-utilization saturation loop...")

    # 4. Continuous High-Load Saturation Loop
    records = []
    t_start_all = time.perf_counter()
    tokens_processed = 0

    # Calculate theoretical FLOPs per sample
    # FLOPs ~ 6 * (24 * L * d^2 * S + 4 * L * S^2 * d)
    L = args.layers
    d = args.d_model
    S = args.seq_len
    flops_per_sample = 6.0 * (24.0 * L * (d**2) * S + 4.0 * L * (S**2) * d)

    print("\n" + "-" * 72)
    print(f" {'Step':<6} | {'Loss':<8} | {'Step (ms)':<10} | {'Throughput':<14} | {'VRAM (GB)':<12} | {'Power (W)':<9}")
    print("-" * 72)

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(input_ids, labels=labels)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_ms = dt * 1000.0

        n_tok = args.batch_size * args.seq_len
        tokens_processed += n_tok
        step_tok_per_sec = n_tok / max(1e-6, dt)
        step_tflops = (args.batch_size * flops_per_sample) / (dt * 1e12)

        telem = get_gpu_telemetry(0)
        vram_str = f"{telem.get('vram_alloc_gb', 0):.2f}/{telem.get('vram_total_gb', 24):.1f} ({telem.get('vram_util_pct', 0)}%)"
        power_str = f"{telem.get('power_watts', 'N/A')}W"

        if step % 5 == 0 or step == 1 or step == args.steps:
            print(f" {step:5d}/{args.steps} | {loss.item():8.4f} | {step_ms:8.2f} ms | {step_tflops:6.1f} TFLOPs | {vram_str:<12} | {power_str:<9}")

        records.append({
            "step": step,
            "loss": round(float(loss.item()), 4),
            "step_ms": round(step_ms, 2),
            "tflops": round(step_tflops, 2),
            "tokens_per_sec": round(step_tok_per_sec, 1),
            "vram_alloc_gb": telem.get("vram_alloc_gb", 0),
            "power_watts": telem.get("power_watts", None),
            "gpu_sm_util_pct": telem.get("gpu_sm_util_pct", None),
        })

    total_time = time.perf_counter() - t_start_all
    avg_tflops = sum(r["tflops"] for r in records) / len(records)
    avg_tok_s = tokens_processed / total_time
    max_vram = max(r["vram_alloc_gb"] for r in records)

    print("-" * 72)
    print("\n======================================================================")
    print("  [OK] AutoMax GPU Stress & Saturation Completed Successfully")
    print("======================================================================")
    print(f"  Total Duration      : {total_time:.2f} s")
    print(f"  Average Throughput  : {avg_tflops:.2f} TFLOPs/sec ({avg_tok_s:,.1f} tokens/s)")
    print(f"  Peak VRAM Allocated : {max_vram:.2f} GB ({max_vram / total_vram_gb * 100:.1f}% saturation)")
    powers = [r["power_watts"] for r in records if r["power_watts"] is not None]
    if powers:
        print(f"  Peak Power Draw     : {max(powers):.1f} Watts (Avg: {sum(powers)/len(powers):.1f} W)")
    print("======================================================================")

    # Save structured telemetry results
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "automax_stress_latest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": device_name,
                "total_vram_gb": total_vram_gb,
                "total_duration_s": round(total_time, 2),
                "avg_tflops": round(avg_tflops, 2),
                "avg_tokens_per_sec": round(avg_tok_s, 1),
                "peak_vram_gb": max_vram,
                "steps": records,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved execution report to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
