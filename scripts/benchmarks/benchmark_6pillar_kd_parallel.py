#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
High-Throughput 6-Pillar Knowledge Distillation Parallel Benchmark Suite.

Executes dedicated 6-pillar knowledge distillation training (up to 2 minutes per module)
supporting concurrent multi-worker execution (e.g. 3 parallel workers on a 24GB GPU)
to fully saturate GPU compute and VRAM capacity.

6 Distillation Pillars:
1. Logit KL Distillation (T=2.0)
2. Hedgehog Feature Map KD (chunked attention map alignment)
3. MiniLM Value-Relation Alignment (head-averaged value Gram matrix)
4. Blockwise Operator Cloning (Frobenius + Cosine distance)
5. Linear CKA Representation Alignment
6. SmoothMax Layer CKA Steering Loss
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops import ATTENTION_REGISTRY
from neuralops.compression import COMPRESSOR_REGISTRY
from neuralops.evaluation.eval_patcher import apply_module_to_model
from neuralops.trainer.distillation.multi_level_kd import (
    BlockwiseOperatorCloningLoss,
    HedgehogFeatureMapKDLoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
    SmoothMaxOverLayersCKASteeringLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("neuralops.kd_6pillar")


class DenseTeacherModel(nn.Module):
    """Standard dense Transformer LM teacher for multi-pillar distillation."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 768, num_heads: int = 12, num_layers: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        feat = self.norm(x)
        logits = self.head(feat)

        # Projections for attention map and relation distillation
        q = feat.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = feat.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = feat.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return logits, hidden_states, (q, k, v)


class SixPillarKDSuite:
    """Coordinates multi-pillar knowledge distillation healing loops."""

    def __init__(self, device: str = "cuda:0", temperature: float = 2.0) -> None:
        self.device = torch.device(device)
        self.l_logit = LogitDistillationLoss(mode="forward_kl", temperature=temperature)
        self.l_feature = HedgehogFeatureMapKDLoss(chunk_size=16, temperature=1.0)
        self.l_relation = MiniLMValueRelationKDLoss()
        self.l_operator = BlockwiseOperatorCloningLoss()
        self.l_cka = LinearCKALoss()
        self.l_steering = SmoothMaxOverLayersCKASteeringLoss()

    def compute_loss(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        labels: torch.Tensor,
        s_hidden: list[torch.Tensor],
        t_hidden: list[torch.Tensor],
        s_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        t_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Pillar 0: Hard Cross-Entropy
        loss_ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))

        # Pillar 1: Logit Distillation
        loss_logit = self.l_logit(s_logits, t_logits.detach())

        # Pillar 2: Feature Map KL (Hedgehog)
        q_t, k_t, _ = t_qkv
        q_s, k_s, v_s = s_qkv
        loss_feat = self.l_feature(q_t.detach(), k_t.detach(), q_s, k_s)

        # Pillar 3: MiniLM Value Relation
        _, _, v_t = t_qkv
        loss_rel = self.l_relation(v_s, v_t.detach())

        # Pillar 4: Blockwise Operator Cloning
        loss_op = self.l_operator(s_hidden[-1], t_hidden[-1].detach())

        # Pillar 5: Linear CKA Alignment
        loss_cka = self.l_cka(s_hidden[-1], t_hidden[-1].detach())

        # Pillar 6: Layer CKA Steering
        loss_steer = self.l_steering(s_hidden, [h.detach() for h in t_hidden])

        # Combined 6-Pillar Objective
        total_loss = (
            0.5 * loss_ce
            + 0.5 * loss_logit
            + 0.1 * loss_feat
            + 0.1 * loss_rel
            + 0.1 * loss_op
            + 0.1 * loss_cka
            + 0.1 * loss_steer
        )

        metrics = {
            "ce": round(loss_ce.item(), 4),
            "logit": round(loss_logit.item(), 4),
            "feat": round(loss_feat.item(), 4),
            "rel": round(loss_rel.item(), 4),
            "op": round(loss_op.item(), 4),
            "cka": round(loss_cka.item(), 4),
            "steer": round(loss_steer.item(), 4),
            "total": round(total_loss.item(), 4),
        }
        return total_loss, metrics


def train_module_kd(
    module_name: str,
    module_type: str,
    teacher: DenseTeacherModel,
    kd_suite: SixPillarKDSuite,
    device: torch.device,
    max_duration_sec: float = 120.0,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 1000,
) -> dict[str, Any]:
    """Train single module under 6-pillar distillation bounded to max_duration_sec."""
    t_start = time.perf_counter()
    student = DenseTeacherModel(vocab_size=vocab_size, hidden_size=768, num_heads=12, num_layers=4)

    try:
        if module_type == "attention":
            student, ratio = apply_module_to_model(student, module_name, module_type="attention", rank=32, device=device)
        else:
            student, ratio = apply_module_to_model(student, module_name, module_type="compressor", rank=32, device=device)
    except Exception as e:
        logger.warning(f"Failed to inject {module_name}: {e}")
        return {
            "module": module_name,
            "category": module_type,
            "status": "INJECTION_FAILED",
            "duration_sec": round(time.perf_counter() - t_start, 2),
            "initial_loss": None,
            "final_loss": None,
            "loss_drop_pct": 0.0,
            "steps_completed": 0,
            "throughput_tokens_per_sec": 0.0,
            "peak_vram_mb": 0.0,
            "params_m": 0.0,
        }

    student.to(device)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)

    steps = 0
    init_loss = None
    last_loss = None
    toks_total = 0

    student.train()
    teacher.eval()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= max_duration_sec:
            break

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            t_logits, t_hidden, t_qkv = teacher(input_ids)

        optimizer.zero_grad(set_to_none=True)
        s_logits, s_hidden, s_qkv = student(input_ids)

        loss, m = kd_suite.compute_loss(s_logits, t_logits, labels, s_hidden, t_hidden, s_qkv, t_qkv)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN loss on module {module_name}; skipping step")
            continue

        if init_loss is None:
            init_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        last_loss = loss.item()
        steps += 1
        toks_total += batch_size * seq_len

    dur = time.perf_counter() - t_start
    tps = toks_total / max(1e-6, dur)
    drop_pct = round(((init_loss - last_loss) / max(1e-6, init_loss)) * 100, 2) if init_loss and last_loss else 0.0

    peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0
    total_params = sum(p.numel() for p in student.parameters())

    del student
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "module": module_name,
        "category": module_type,
        "status": "PASS",
        "duration_sec": round(dur, 2),
        "initial_loss": round(init_loss, 4) if init_loss else 0.0,
        "final_loss": round(last_loss, 4) if last_loss else 0.0,
        "loss_drop_pct": drop_pct,
        "steps_completed": steps,
        "throughput_tokens_per_sec": round(tps, 1),
        "peak_vram_mb": peak_vram,
        "params_m": round(total_params / 1e6, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="6-Pillar KD Parallel Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--worker-id", type=int, default=0, help="Worker rank (0, 1, 2)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total workers (e.g. 3)")
    parser.add_argument("--output-csv", type=str, default="results/csv/kd_6pillar_worker0.csv")
    args = parser.parse_args()

    dev = torch.device(args.device)
    teacher = DenseTeacherModel().to(dev)
    kd_suite = SixPillarKDSuite(device=args.device)

    all_modules: list[tuple[str, str]] = []
    for k in sorted(ATTENTION_REGISTRY.keys()):
        all_modules.append((k, "attention"))
    for k in sorted(COMPRESSOR_REGISTRY.keys()):
        all_modules.append((k, "compressor"))

    # Shard modules for this worker
    worker_modules = [m for idx, m in enumerate(all_modules) if idx % args.num_workers == args.worker_id]
    logger.info(f"Worker [{args.worker_id}/{args.num_workers}] assigned {len(worker_modules)} modules on {args.device}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "module", "category", "status", "duration_sec", "initial_loss",
        "final_loss", "loss_drop_pct", "steps_completed", "throughput_tokens_per_sec",
        "peak_vram_mb", "params_m"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for idx, (m_name, m_type) in enumerate(worker_modules, 1):
            logger.info(f"[Worker {args.worker_id}] [{idx}/{len(worker_modules)}] Starting 6-Pillar KD: {m_name} ({m_type})...")
            res = train_module_kd(m_name, m_type, teacher, kd_suite, dev, max_duration_sec=args.max_duration)
            writer.writerow(res)
            f.flush()
            logger.info(
                f"[Worker {args.worker_id}]   -> Status: {res['status']} | Steps: {res['steps_completed']} | "
                f"Loss: {res['initial_loss']} -> {res['final_loss']} ({res['loss_drop_pct']}%) | "
                f"Throughput: {res['throughput_tokens_per_sec']} tok/s"
            )

    logger.info(f"[Worker {args.worker_id}] Finished all assigned modules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
