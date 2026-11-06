#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
High-Load GPU Heavy Training & Stress Benchmark.

Saturates GPU compute and VRAM on modern accelerators (e.g. NVIDIA RTX 3090 24GB):
- 12-layer multi-head transformer with 1024 hidden dimension and 16 heads (~150M params)
- FP16 / BF16 mixed-precision training with gradient scaling
- Multi-pillar knowledge distillation healing objective + cross-entropy loss
- Warmup-Stable-Decay (WSD) scheduler
- Persistent checkpointing and real-time telemetry logging
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralops


class TransformerEncoderBlock(nn.Module):
    """Production transformer encoder block with self-attention and MLP."""

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16, intermediate_dim: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        norm_x = self.norm1(x)

        q = self.q_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product attention (handled by PyTorch or compat shim)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.out_proj(attn)

        # MLP residual
        x = x + self.mlp(self.norm2(x))
        return x


class HeavyTransformerModel(nn.Module):
    """Multi-layer heavy transformer model for GPU stress training."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_dim: int = 4096,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding[:, :S, :]
        for layer in self.layers:
            x = layer(x)
        features = self.norm(x)
        logits = self.classifier(features.mean(dim=1))
        return logits, features


def get_wsd_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float = 0.1, stable_ratio: float = 0.7) -> float:
    """Warmup-Stable-Decay (WSD) learning rate schedule."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    stable_steps = int(total_steps * stable_ratio)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    elif step < stable_steps:
        return base_lr
    else:
        decay_progress = (step - stable_steps) / max(1, total_steps - stable_steps)
        return max(1e-7, base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress)))


def run_heavy_stress_training(
    steps: int = 1000,
    batch_size: int = 16,
    seq_len: int = 256,
    hidden_dim: int = 768,
    num_layers: int = 8,
    num_heads: int = 12,
    accum_steps: int = 1,
    base_lr: float = 5e-4,
    checkpoint_dir: str = "results/checkpoints",
    device: str = "cuda",
) -> None:
    """Execute high-throughput GPU stress training loop."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(device)
    print("=" * 75)
    print("  [*] NeuralOps Heavy GPU Stress Training & Workload Engine")
    print(f"  Target Device     : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM Capacity     : {vram_total:.2f} GB")
    print(f"  Model Arch        : {num_layers} layers | {hidden_dim} dim | {num_heads} heads")
    print(f"  Batch Spec        : Batch Size {batch_size} | Seq Len {seq_len} | Total Steps {steps}")
    print("=" * 75)

    # Initialize Student and Teacher models for Knowledge Distillation
    print("Initializing Student and Teacher transformer architectures...", flush=True)
    student = HeavyTransformerModel(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads).to(device)
    teacher = HeavyTransformerModel(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    param_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  [+] Model initialized with {param_count / 1e6:.2f}M trainable parameters.", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=base_lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Verify checkpoint directory
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    ce_criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss(reduction="batchmean")
    mse_criterion = nn.MSELoss()

    vocab_size = 30522
    t_start = time.perf_counter()
    tokens_processed = 0

    print("Beginning stress execution loop...\n", flush=True)
    for step in range(steps):
        t_step0 = time.perf_counter()
        lr = get_wsd_lr(step, steps, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Generate synthetic input batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device)

        # Forward pass with mixed precision
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                student_logits, student_feat = student(input_ids)
                with torch.no_grad():
                    teacher_logits, teacher_feat = teacher(input_ids)

                loss_ce = ce_criterion(student_logits, targets)
                loss_kd = kd_criterion(
                    F.log_softmax(student_logits / 2.0, dim=-1),
                    F.softmax(teacher_logits / 2.0, dim=-1),
                ) * (2.0 ** 2)
                loss_feat = mse_criterion(student_feat, teacher_feat)
                total_loss = (0.5 * loss_ce + 0.4 * loss_kd + 0.1 * loss_feat) / accum_steps
        else:
            student_logits, student_feat = student(input_ids)
            with torch.no_grad():
                teacher_logits, teacher_feat = teacher(input_ids)
            loss_ce = ce_criterion(student_logits, targets)
            loss_kd = kd_criterion(
                F.log_softmax(student_logits / 2.0, dim=-1),
                F.softmax(teacher_logits / 2.0, dim=-1),
            ) * (2.0 ** 2)
            loss_feat = mse_criterion(student_feat, teacher_feat)
            total_loss = (0.5 * loss_ce + 0.4 * loss_kd + 0.1 * loss_feat) / accum_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            total_loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        step_time_ms = (time.perf_counter() - t_step0) * 1000
        tokens_in_step = batch_size * seq_len
        tokens_processed += tokens_in_step
        throughput = tokens_in_step / max(1e-4, step_time_ms / 1000)

        # Telemetry logging every 20 steps
        if (step + 1) % 20 == 0 or step == steps - 1:
            vram_info = ""
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                vram_res = torch.cuda.memory_reserved() / 1e9
                vram_info = f" | VRAM: {vram_alloc:.2f}/{vram_res:.2f} GB"

            print(
                f"  [Step {step+1:4d}/{steps:4d}] "
                f"Loss: {total_loss.item() * accum_steps:.4f} "
                f"(CE: {loss_ce.item():.3f}, KD: {loss_kd.item():.3f}) | "
                f"LR: {lr:.2e} | "
                f"Step: {step_time_ms:.1f}ms | "
                f"Throughput: {throughput:6.0f} tok/s"
                f"{vram_info}",
                flush=True,
            )

        # Checkpointing
        if (step + 1) % 250 == 0:
            ckpt_file = ckpt_path / f"stress_checkpoint_step_{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state": student.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": total_loss.item() * accum_steps,
                },
                ckpt_file,
            )
            print(f"  [SAVED] Saved stress checkpoint to {ckpt_file}", flush=True)

    total_duration = time.perf_counter() - t_start
    overall_throughput = tokens_processed / max(1e-4, total_duration)
    print("\n" + "=" * 75)
    print("  [SUCCESS] Heavy GPU Stress Training Completed Successfully!")
    print(f"  Total Duration    : {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"  Total Tokens      : {tokens_processed:,}")
    print(f"  Mean Throughput   : {overall_throughput:.1f} tokens/second")
    print("=" * 75, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuralOps Heavy GPU Stress Training Workload")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per forward pass (default: 16)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension (default: 768)")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer layers (default: 8)")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads (default: 12)")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)")
    parser.add_argument("--device", default="cuda", help="Execution device; CUDA is required by default")
    args = parser.parse_args()

    run_heavy_stress_training(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        accum_steps=args.accum_steps,
        base_lr=args.lr,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
