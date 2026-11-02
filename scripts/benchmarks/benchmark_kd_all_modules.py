#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Per-Module 2-Minute Knowledge Distillation (KD) Benchmark Suite.

Executes a dedicated, bounded knowledge distillation training session (up to 2 minutes per module)
for each registered Attention Mechanism and Compression Engine on GPU hardware.
Records live metrics (Loss convergence, CE loss, KD loss, Throughput, VRAM footprint, Parameter count)
and atomically persists structured results to SQLite master ledger, CSV, and summary reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

# Ensure neuralops is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, create_attention
from neuralops.trainer.distillation.multi_level_kd import LogitDistillationLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("neuralops.kd_benchmark")


class SimpleTransformerLM(nn.Module):
    """Clean transformer language model supporting arbitrary injected attention layers."""

    def __init__(
        self,
        attn_layer: nn.Module,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attn_layer
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        attn_tensor = attn_out[0] if isinstance(attn_out, (tuple, list)) else attn_out
        x = x + attn_tensor
        x = x + self.mlp(self.norm2(x))
        return self.head(x)


class StandardTeacherLM(nn.Module):
    """Dense standard multi-head attention teacher model for distillation."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x_norm = self.norm1(x)
        mha_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + mha_out
        x = x + self.mlp(self.norm2(x))
        return self.head(x)


def run_single_kd_session(
    module_name: str,
    module_type: str,
    target_module: nn.Module,
    teacher_model: nn.Module,
    device: torch.device,
    max_duration_sec: float = 120.0,
    batch_size: int = 8,
    seq_len: int = 64,
    vocab_size: int = 1000,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    """Train student model with knowledge distillation for at most max_duration_sec."""
    if module_type == "attention":
        student_model = SimpleTransformerLM(target_module, vocab_size=vocab_size, embed_dim=128).to(device)
    else:
        student_model = target_module.to(device)

    student_model.train()
    teacher_model.eval()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0.01)
    kd_loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=2.0)

    total_tokens = 0
    step = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    loss_history: list[float] = []

    start_time = time.perf_counter()
    has_cuda = (device.type == "cuda")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    param_count = sum(p.numel() for p in student_model.parameters())

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= max_duration_sec:
            break

        # Generate synthetic dynamic token batch
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.no_grad():
                t_logits = teacher_model(inputs)

            s_logits = student_model(inputs)
            s_logits_flat = s_logits.view(-1, vocab_size)

            ce_loss = F.cross_entropy(s_logits_flat, targets)
            kd_loss = kd_loss_fn(s_logits_flat, t_logits.view(-1, vocab_size))
            total_loss = ce_loss + 0.5 * kd_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("[%s] NaN/Inf loss encountered at step %d, terminating early", module_name, step)
                break

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            loss_val = float(total_loss.item())
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val
            loss_history.append(loss_val)

            total_tokens += batch_size * seq_len
            step += 1

        except Exception as exc:
            logger.warning("[%s] Training exception: %s", module_name, exc)
            return {
                "module_name": module_name,
                "module_type": module_type,
                "status": "ERROR",
                "error": str(exc),
                "steps_completed": step,
                "duration_sec": round(elapsed, 2),
                "initial_loss": initial_loss or 0.0,
                "final_loss": final_loss or 0.0,
                "loss_reduction": 0.0,
                "throughput_tok_s": 0.0,
                "peak_vram_mb": 0.0,
                "param_count": param_count,
            }

    duration = time.perf_counter() - start_time
    throughput = total_tokens / max(duration, 1e-6)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if has_cuda else 0.0
    loss_reduction = round((initial_loss - final_loss) / max(abs(initial_loss), 1e-6) * 100.0, 2) if (initial_loss and final_loss) else 0.0

    return {
        "module_name": module_name,
        "module_type": module_type,
        "status": "PASS",
        "error": "",
        "steps_completed": step,
        "duration_sec": round(duration, 2),
        "initial_loss": round(initial_loss or 0.0, 4),
        "final_loss": round(final_loss or 0.0, 4),
        "loss_reduction": loss_reduction,
        "throughput_tok_s": round(throughput, 1),
        "peak_vram_mb": round(peak_vram, 2),
        "param_count": param_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive Per-Module 2-Minute KD Benchmark")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-duration", type=float, default=120.0, help="Max duration per module in seconds (default: 120s = 2 min)")
    parser.add_argument("--output-csv", type=Path, default=_REPO_ROOT / "results" / "csv" / "kd_per_module_benchmark.csv")
    parser.add_argument("--output-sqlite", type=Path, default=_REPO_ROOT / "results" / "master_ledger.sqlite")
    parser.add_argument("--category", choices=["attention", "compressor", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke testing")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Initializing KD benchmark on device: %s (Max duration per module: %.1fs)", device, args.max_duration)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Initialize frozen teacher model
    teacher_model = StandardTeacherLM(vocab_size=1000, embed_dim=128, num_heads=4).to(device)
    teacher_model.eval()

    results: list[dict[str, Any]] = []

    # 1. Attention Mechanisms
    if args.category in ("attention", "all"):
        ATTENTION_REGISTRY._load_all()
        attn_keys = sorted(list(ATTENTION_REGISTRY._dict.keys()))
        if args.limit:
            attn_keys = attn_keys[:args.limit]
        logger.info("Starting Knowledge Distillation runs for %d attention mechanisms...", len(attn_keys))

        for idx, name in enumerate(attn_keys, 1):
            logger.info("[%d/%d] Training Attention KD: %s (up to %.0fs)...", idx, len(attn_keys), name, args.max_duration)
            try:
                attn_layer = create_attention(name, embed_dim=128, num_heads=4)
                rec = run_single_kd_session(
                    name, "attention", attn_layer, teacher_model, device,
                    max_duration_sec=args.max_duration
                )
            except Exception as e:
                rec = {
                    "module_name": name,
                    "module_type": "attention",
                    "status": "ERROR",
                    "error": str(e),
                    "steps_completed": 0,
                    "duration_sec": 0.0,
                    "initial_loss": 0.0,
                    "final_loss": 0.0,
                    "loss_reduction": 0.0,
                    "throughput_tok_s": 0.0,
                    "peak_vram_mb": 0.0,
                    "param_count": 0,
                }
            results.append(rec)
            logger.info("  -> Status: %s | Steps: %d | Loss: %.4f -> %.4f (%.1f%%) | Throughput: %.1f tok/s",
                        rec["status"], rec["steps_completed"], rec["initial_loss"], rec["final_loss"],
                        rec["loss_reduction"], rec["throughput_tok_s"])

    # 2. Compressor Engines
    if args.category in ("compressor", "all"):
        COMPRESSOR_REGISTRY._load_all()
        comp_keys = sorted(list(COMPRESSOR_REGISTRY._dict.keys()))
        if args.limit:
            comp_keys = comp_keys[:args.limit]
        logger.info("Starting Knowledge Distillation runs for %d compression engines...", len(comp_keys))

        for idx, name in enumerate(comp_keys, 1):
            logger.info("[%d/%d] Training Compressor KD: %s (up to %.0fs)...", idx, len(comp_keys), name, args.max_duration)
            try:
                cls = COMPRESSOR_REGISTRY.get(name)
                from transformers import BertConfig, BertModel
                cfg = BertConfig(num_hidden_layers=1, hidden_size=64, num_attention_heads=2, intermediate_size=128)
                base_model = BertModel(cfg)
                if hasattr(cls, "compress"):
                    comp = cls(model=base_model, rank=8, fallback_to_original=False)
                    res = comp.compress()
                    comp_model = res[0] if isinstance(res, tuple) else res
                elif callable(cls):
                    try:
                        res = cls(base_model, rank=8, fallback_to_original=False)
                    except TypeError:
                        res = cls(model=base_model, rank=8)
                    comp_model = res[0] if isinstance(res, tuple) else (res.compress()[0] if hasattr(res, "compress") else res)
                else:
                    raise RuntimeError(f"Compressor {name} is not callable")

                rec = run_single_kd_session(
                    name, "compressor", comp_model, teacher_model, device,
                    max_duration_sec=args.max_duration, seq_len=32
                )
            except Exception as e:
                rec = {
                    "module_name": name,
                    "module_type": "compressor",
                    "status": "ERROR",
                    "error": str(e),
                    "steps_completed": 0,
                    "duration_sec": 0.0,
                    "initial_loss": 0.0,
                    "final_loss": 0.0,
                    "loss_reduction": 0.0,
                    "throughput_tok_s": 0.0,
                    "peak_vram_mb": 0.0,
                    "param_count": 0,
                }
            results.append(rec)
            logger.info("  -> Status: %s | Steps: %d | Loss: %.4f -> %.4f (%.1f%%) | Throughput: %.1f tok/s",
                        rec["status"], rec["steps_completed"], rec["initial_loss"], rec["final_loss"],
                        rec["loss_reduction"], rec["throughput_tok_s"])

    # Persist results to CSV
    fieldnames = [
        "module_name", "module_type", "status", "steps_completed", "duration_sec",
        "initial_loss", "final_loss", "loss_reduction", "throughput_tok_s", "peak_vram_mb", "param_count", "error"
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info("Saved %d KD benchmark results to %s", len(results), args.output_csv)

    # Persist to SQLite Master Ledger
    try:
        conn = sqlite3.connect(args.output_sqlite)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS per_module_kd_benchmarks (
                module_name TEXT,
                module_type TEXT,
                status TEXT,
                steps_completed INTEGER,
                duration_sec REAL,
                initial_loss REAL,
                final_loss REAL,
                loss_reduction REAL,
                throughput_tok_s REAL,
                peak_vram_mb REAL,
                param_count INTEGER,
                error TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (module_name, module_type)
            )
        """)
        for r in results:
            cursor.execute("""
                INSERT OR REPLACE INTO per_module_kd_benchmarks (
                    module_name, module_type, status, steps_completed, duration_sec,
                    initial_loss, final_loss, loss_reduction, throughput_tok_s,
                    peak_vram_mb, param_count, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["module_name"], r["module_type"], r["status"], r["steps_completed"], r["duration_sec"],
                r["initial_loss"], r["final_loss"], r["loss_reduction"], r["throughput_tok_s"],
                r["peak_vram_mb"], r["param_count"], r["error"]
            ))
        conn.commit()
        conn.close()
        logger.info("Atomically recorded %d entries into SQLite master ledger at %s", len(results), args.output_sqlite)
    except Exception as e:
        logger.warning("Failed to write to SQLite master ledger: %s", e)

    # Generate Markdown Summary
    report_md = _REPO_ROOT / "results" / "reports" / "KD_PER_MODULE_REPORT.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# 🧠 Comprehensive Per-Module Knowledge Distillation (KD) Benchmark\n\n")
        f.write(f"- **Total Modules Tested:** `{len(results)}`\n")
        passed = sum(1 for r in results if r["status"] == "PASS")
        f.write(f"- **Passed Runs:** `{passed}/{len(results)}` (`{round(passed/max(1, len(results))*100, 1)}%`)\n")
        f.write(f"- **Max Duration Per Module:** `{args.max_duration:.0f}s` (2 minutes limit)\n\n")
        f.write("## Detailed Module Breakdown\n\n")
        f.write("| Module | Type | Status | Steps | Duration (s) | Loss Drop | Throughput (tok/s) | Peak VRAM (MB) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            status_icon = "✅" if r["status"] == "PASS" else "❌"
            f.write(f"| `{r['module_name']}` | `{r['module_type']}` | {status_icon} `{r['status']}` | {r['steps_completed']} | {r['duration_sec']}s | `{r['initial_loss']:.4f} -> {r['final_loss']:.4f}` ({r['loss_reduction']}%) | {r['throughput_tok_s']} | {r['peak_vram_mb']} MB |\n")
    logger.info("Generated markdown report at %s", report_md)


if __name__ == "__main__":
    main()
