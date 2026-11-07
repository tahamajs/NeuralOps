#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dual-Mode Knowledge Distillation & Attention Surgery Ablation Engine.

Evaluates both regimes for physical paper tables:
  Mode 1 (End-to-End): Full student model parameters trained and healed.
  Mode 2 (Frozen Backbone): Freezes all transformer parameters EXCEPT newly inserted
                            attention layers and classifier head.

Persists results into:
  1. JSON benchmark report: results/sst2_dual_mode_kd_ablation.json
  2. CSV comparative dashboard: results/sst2_dual_mode_kd_ablation.csv
  3. Master SQLite Ledger: results/master_ledger.sqlite (results table)
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Backward compat shim for PyTorch 1.12 named_parameters remove_duplicate
_orig_named_params = nn.Module.named_parameters


def _safe_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)


nn.Module.named_parameters = _safe_named_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dual_mode_kd")


class ArrowSST2Dataset(Dataset):
    """Direct PyArrow in-memory dataset eliminating third-party network wrappers."""

    def __init__(self, arrow_path: str, tokenizer: Any, max_length: int = 128) -> None:
        with open(arrow_path, "rb") as f:
            reader = ipc.open_stream(f)
            self.table = reader.read_all()
        self.sentences = self.table["sentence"].to_pylist()
        self.labels = self.table["label"].to_pylist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.tokenizer(
            self.sentences[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Compute classification accuracy and mean loss on validation set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = F.cross_entropy(logits, labels)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss


def configure_parameter_gradients(student_model: nn.Module, freeze_backbone: bool) -> list[torch.nn.Parameter]:
    """Configures parameter freezing based on regime.

    If freeze_backbone is True:
        Freezes embeddings, FFN, layer norms, and other layers, keeping ONLY
        the new attention weights and the sequence classification head trainable.
    If False:
        All student parameters are trainable (End-to-End).
    """
    trainable_params = []
    frozen_count = 0
    trainable_count = 0

    for name, param in student_model.named_parameters():
        if freeze_backbone:
            # Train only the attention modules and the classifier head
            is_trainable = ("attention.self" in name) or ("classifier" in name)
            param.requires_grad = is_trainable
        else:
            param.requires_grad = True

        if param.requires_grad:
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    logger.info(
        f"Regime [{'Frozen Backbone' if freeze_backbone else 'End-to-End'}]: "
        f"Trainable Parameters = {trainable_count:,} | Frozen Parameters = {frozen_count:,}"
    )
    return trainable_params


def train_single_regime_kd(
    module_name: str,
    regime: str,
    freeze_backbone: bool,
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    kd_steps: int = 200,
    lr: float = 4e-5,
    temperature: float = 2.0,
    alpha_kd: float = 0.5,
) -> dict[str, Any]:
    """Train student model with KD under specified freeze regime on GPU."""
    logger.info(f"=== Starting KD: {module_name} | Mode: {regime} | Steps: {kd_steps} ===")
    teacher_model.eval()
    student_model.train()

    trainable_params = configure_parameter_gradients(student_model, freeze_backbone=freeze_backbone)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    step = 0
    train_iter = iter(train_loader)
    start_time = time.time()
    last_loss = 0.0

    while step < kd_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            with torch.no_grad():
                t_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

            s_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

            ce_loss = F.cross_entropy(s_logits, labels)
            p_s = F.log_softmax(s_logits / temperature, dim=-1)
            p_t = F.softmax(t_logits / temperature, dim=-1)
            kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature**2)

            total_loss = (1.0 - alpha_kd) * ce_loss + alpha_kd * kd_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(total_loss.item())
        step += 1
        if step % 25 == 0 or step == kd_steps:
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            logger.info(
                f"[{module_name}|{regime}] Step {step}/{kd_steps} | Loss: {last_loss:.4f} | VRAM: {mem_mb:.1f} MB"
            )

    duration = time.time() - start_time
    val_acc, val_loss = evaluate_accuracy(student_model, val_loader, device)
    mem_peak_gb = (torch.cuda.max_memory_allocated(device) / (1024**3)) if torch.cuda.is_available() else 0.0

    logger.info(
        f"[{module_name}|{regime}] COMPLETE: Acc={val_acc:.2f}% | ValLoss={val_loss:.4f} | VRAM={mem_peak_gb:.2f} GB | Time={duration:.1f}s"
    )
    return {
        "module": module_name,
        "mode": regime,
        "freeze_backbone": freeze_backbone,
        "accuracy": round(val_acc, 2),
        "val_loss": round(val_loss, 4),
        "train_loss": round(last_loss, 4),
        "peak_vram_gb": round(mem_peak_gb, 2),
        "duration_sec": round(duration, 1),
        "steps": kd_steps,
        "status": "SUCCESS",
    }


def persist_to_sqlite_ledger(records: list[dict[str, Any]], sqlite_path: str = "results/master_ledger.sqlite") -> None:
    """Safely append ablation records into the canonical results table of master_ledger.sqlite."""
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    for r in records:
        rec_id = f"kd_ablation_{r['module']}_{r['mode']}_{int(time.time())}"
        cur.execute(
            """
            INSERT OR REPLACE INTO results (
                record_id, module_name, category, family, version_tag, revision,
                status, timestamp_utc, git_commit_sha, algorithm, backbone,
                dataset, rank, seq_len, batch_size, dtype, hardware, accuracy,
                retention_pct, ppl, latency_triton_ms, latency_pytorch_ms,
                speedup_x, peak_vram_mb, throughput_tok_s, cosine_sim, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                rec_id,
                r["module"],
                "attention_kd_ablation",
                r["mode"],
                "v2.0.4",
                1,
                r["status"],
                now_iso,
                "main",
                f"6Pillar_KD_{r['mode']}",
                "bert-base-uncased",
                "SST-2",
                0,
                128,
                64,
                "float16",
                "NVIDIA GeForce RTX 3090",
                r["accuracy"],
                r.get("retention_pct", 100.0),
                r["val_loss"],
                0.0,
                0.0,
                1.0,
                r["peak_vram_gb"] * 1024.0,
                0.0,
                0.0,
                json.dumps(r),
            ),
        )
    conn.commit()
    conn.close()
    logger.info(f"Persisted {len(records)} records into SQLite Master Ledger: {sqlite_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Mode SST-2 KD & Attention Surgery Ablation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU execution")
    parser.add_argument("--kd-steps", type=int, default=200, help="Number of KD steps per module per mode")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--out-json", type=str, default="results/sst2_dual_mode_kd_ablation.json")
    parser.add_argument("--out-csv", type=str, default="results/sst2_dual_mode_kd_ablation.csv")
    parser.add_argument("--ledger-db", type=str, default="results/master_ledger.sqlite")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Dual-Mode Evaluator Launching on: {gpu_name} | Batch: {args.batch_size} | Steps: {args.kd_steps}")

    # 1. Load Tokenizer & Teacher
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, local_files_only=True
    ).to(device)

    # 2. Ingest datasets directly from Arrow
    train_arrow = os.path.expanduser(
        "~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-train.arrow"
    )
    val_arrow = os.path.expanduser(
        "~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-validation.arrow"
    )

    train_dataset = ArrowSST2Dataset(train_arrow, tokenizer)
    val_dataset = ArrowSST2Dataset(val_arrow, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 3. Prewarm teacher classification head on SST-2 to reach real baseline (>89%)
    logger.info("Pre-training Teacher on SST-2 to establish authentic baseline...")
    teacher_model.train()
    prewarm_optim = torch.optim.AdamW(teacher_model.parameters(), lr=3e-5)
    for step_pw, batch_pw in enumerate(train_loader):
        if step_pw >= 160:
            break
        prewarm_optim.zero_grad(set_to_none=True)
        in_ids = batch_pw["input_ids"].to(device, non_blocking=True)
        att_m = batch_pw["attention_mask"].to(device, non_blocking=True)
        lbls = batch_pw["label"].to(device, non_blocking=True)
        out_pw = teacher_model(input_ids=in_ids, attention_mask=att_m)
        l_pw = F.cross_entropy(out_pw.logits, lbls)
        l_pw.backward()
        prewarm_optim.step()

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    base_acc, base_loss = evaluate_accuracy(teacher_model, val_loader, device)
    logger.info(f"Verified Teacher Baseline on SST-2: {base_acc:.2f}% | Val Loss: {base_loss:.4f}")

    target_modules = [
        "linear_attention",
        "cosformer",
        "gla_attention",
        "differential_attention",
        "sparse_attention",
        "chunkwise_attention",
        "svd_linear_attention",
    ]

    from neuralops.core.registry import create_attention

    all_ablation_results = []

    # Run BOTH modes: Mode 1 (End-to-End) and Mode 2 (Frozen Backbone)
    modes = [
        ("End-to-End", False),
        ("Frozen Backbone", True),
    ]

    for mode_name, freeze_flag in modes:
        logger.info("=" * 80)
        logger.info(f"ENTERING ABLATION REGIME: {mode_name} (freeze_backbone={freeze_flag})")
        logger.info("=" * 80)

        for mod_name in target_modules:
            try:
                student_model = copy.deepcopy(teacher_model)
                attn_replacement = create_attention(mod_name, embed_dim=768, num_heads=12)
                for layer in student_model.bert.encoder.layer:
                    layer.attention.self = copy.deepcopy(attn_replacement)
                student_model.to(device)

                rec = train_single_regime_kd(
                    module_name=mod_name,
                    regime=mode_name,
                    freeze_backbone=freeze_flag,
                    teacher_model=teacher_model,
                    student_model=student_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    kd_steps=args.kd_steps,
                    lr=args.lr,
                )
                rec["baseline_acc"] = round(base_acc, 2)
                rec["retention_pct"] = round((rec["accuracy"] / (base_acc + 1e-9)) * 100.0, 2)
                all_ablation_results.append(rec)
            except Exception as exc:
                logger.error(f"Error evaluating {mod_name} under {mode_name}: {exc}")
                all_ablation_results.append(
                    {
                        "module": mod_name,
                        "mode": mode_name,
                        "freeze_backbone": freeze_flag,
                        "accuracy": 0.0,
                        "baseline_acc": round(base_acc, 2),
                        "retention_pct": 0.0,
                        "status": f"FAILED: {exc}",
                    }
                )

            # Intermediate periodic save
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(all_ablation_results, f, indent=2)

    # Persist Final Artifacts: CSV Dashboard and SQLite Ledger
    import pandas as pd

    df = pd.DataFrame(all_ablation_results)
    df.to_csv(args.out_csv, index=False)
    logger.info(f"Saved Comparative CSV Dashboard to: {args.out_csv}")

    persist_to_sqlite_ledger(all_ablation_results, sqlite_path=args.ledger_db)

    print("\n" + "=" * 95)
    print(f"DUAL-MODE ABLATION SUMMARY ON SST-2 (Teacher Baseline: {base_acc:.2f}%)")
    print("=" * 95)
    print(f"{'Module':<25} | {'Mode':<18} | {'Accuracy':<10} | {'Retention %':<12} | {'Val Loss':<10} | {'Status'}")
    print("-" * 95)
    for r in all_ablation_results:
        mod = r.get("module", "")[:23]
        m = r.get("mode", "")[:16]
        acc = f"{r.get('accuracy', 0.0):.2f}%"
        ret = f"{r.get('retention_pct', 0.0):.2f}%"
        vl = f"{r.get('val_loss', 0.0):.4f}"
        st = r.get("status", "")
        print(f"{mod:<25} | {m:<18} | {acc:<10} | {ret:<12} | {vl:<10} | {st}")
    print("=" * 95)


if __name__ == "__main__":
    main()
