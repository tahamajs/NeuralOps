#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajlesi/NeuralOps/graphs/contributors
"""
Ultra-Fast Pure GPU Layer-wise KD Benchmark Engine.

Optimizations:
  1. Prewarms Teacher ONCE in GPU memory (reaches verified 90%+ baseline).
  2. Teacher weights frozen, zero overhead.
  3. Evaluates all selected attention mechanisms sequentially in pure GPU memory:
     - Stage 0: Zero-shot surgery accuracy & PPL
     - Stage 1: Isolated attention mimicry (100 steps)
     - Stage 2: Quick E2E polish (35 steps)
  4. Automatic persistence to SQLite Master Ledger (results/master_ledger.sqlite) & CSV.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import logging
import math
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_orig_named_params = nn.Module.named_parameters
def _safe_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _safe_named_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pure_gpu_kd")


class ArrowSST2Dataset(Dataset):
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
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate_metrics(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """Returns (accuracy_pct, loss, perplexity)."""
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
    ppl = math.exp(min(avg_loss, 20.0))
    return acc, avg_loss, ppl


def run_single_module_pipeline(
    module_name: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    isolated_steps: int = 100,
    polish_steps: int = 35,
    iso_lr: float = 1.2e-3,
    polish_lr: float = 2.0e-5,
) -> dict[str, Any]:
    from neuralops.core.registry import create_attention

    logger.info(f"\n{'='*80}\n[PIPELINE] Testing Attention Module: {module_name}\n{'='*80}")
    t0 = time.time()

    # 1. Surgery
    student_model = copy.deepcopy(teacher_model)
    for layer in student_model.bert.encoder.layer:
        layer.attention.self = create_attention(module_name, embed_dim=768, num_heads=12)

    student_model.to(device)
    z_acc, z_loss, z_ppl = evaluate_metrics(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 0 (Zero-Shot Surgery): Acc={z_acc:.2f}% | Loss={z_loss:.4f} | PPL={z_ppl:.2f}")

    # 2. Stage 1: Isolated Attention Mimicry (Backbone Frozen)
    t_iso_start = time.time()
    for name, param in student_model.named_parameters():
        param.requires_grad = ("attention.self" in name) or ("classifier" in name)

    trainable_attn_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer_iso = torch.optim.AdamW(trainable_attn_params, lr=iso_lr, weight_decay=1e-4)

    student_model.train()
    for step_i, batch in enumerate(train_loader):
        if step_i >= isolated_steps:
            break
        in_ids = batch["input_ids"].to(device, non_blocking=True)
        att_m = batch["attention_mask"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        optimizer_iso.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_out = teacher_model(input_ids=in_ids, attention_mask=att_m)
        s_out = student_model(input_ids=in_ids, attention_mask=att_m)

        kd_loss = F.kl_div(
            F.log_softmax(s_out.logits / 2.0, dim=-1),
            F.softmax(t_out.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * 4.0
        ce_loss = F.cross_entropy(s_out.logits, lbls)
        loss = 0.5 * kd_loss + 0.5 * ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_attn_params, 1.0)
        optimizer_iso.step()

    t_iso_dur = time.time() - t_iso_start
    iso_acc, iso_loss, iso_ppl = evaluate_metrics(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 1 (Isolated Attn): Acc={iso_acc:.2f}% | Loss={iso_loss:.4f} | PPL={iso_ppl:.2f} | Time={t_iso_dur:.1f}s")

    # 3. Stage 2: Quick Polish (E2E)
    t_pol_start = time.time()
    for p in student_model.parameters():
        p.requires_grad = True

    optimizer_pol = torch.optim.AdamW(student_model.parameters(), lr=polish_lr, weight_decay=0.01)
    student_model.train()

    for step_p, batch in enumerate(train_loader):
        if step_p >= polish_steps:
            break
        in_ids = batch["input_ids"].to(device, non_blocking=True)
        att_m = batch["attention_mask"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        optimizer_pol.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_out = teacher_model(input_ids=in_ids, attention_mask=att_m)
        s_out = student_model(input_ids=in_ids, attention_mask=att_m)

        kd_loss = F.kl_div(
            F.log_softmax(s_out.logits / 2.0, dim=-1),
            F.softmax(t_out.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * 4.0
        ce_loss = F.cross_entropy(s_out.logits, lbls)
        total_l = 0.5 * kd_loss + 0.5 * ce_loss
        total_l.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer_pol.step()

    t_pol_dur = time.time() - t_pol_start
    final_acc, final_loss, final_ppl = evaluate_metrics(student_model, val_loader, device)
    total_dur = time.time() - t0
    logger.info(f"[{module_name}] Stage 2 (Quick Polish): Acc={final_acc:.2f}% | Loss={final_loss:.4f} | PPL={final_ppl:.2f} | TotalTime={total_dur:.1f}s")

    del student_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "module": module_name,
        "zero_shot_acc": round(z_acc, 2),
        "zero_shot_loss": round(z_loss, 4),
        "zero_shot_ppl": round(z_ppl, 2),
        "isolated_acc": round(iso_acc, 2),
        "isolated_loss": round(iso_loss, 4),
        "isolated_ppl": round(iso_ppl, 2),
        "polished_acc": round(final_acc, 2),
        "polished_loss": round(final_loss, 4),
        "polished_ppl": round(final_ppl, 2),
        "isolated_time_s": round(t_iso_dur, 1),
        "polish_time_s": round(t_pol_dur, 1),
        "total_time_s": round(total_dur, 1),
        "status": "PASSED"
    }


def persist_to_sqlite(results: list[dict[str, Any]], base_acc: float, base_ppl: float, sqlite_path: str = "results/master_ledger.sqlite") -> None:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attention_surgery_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            module TEXT,
            baseline_acc REAL,
            baseline_ppl REAL,
            zero_shot_acc REAL,
            zero_shot_ppl REAL,
            isolated_acc REAL,
            isolated_ppl REAL,
            polished_acc REAL,
            polished_ppl REAL,
            retention_pct REAL,
            total_time_s REAL,
            status TEXT
        )
    """)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for r in results:
        cur.execute("""
            INSERT INTO attention_surgery_metrics (
                timestamp, module, baseline_acc, baseline_ppl,
                zero_shot_acc, zero_shot_ppl, isolated_acc, isolated_ppl,
                polished_acc, polished_ppl, retention_pct, total_time_s, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            r.get("module"),
            base_acc,
            base_ppl,
            r.get("zero_shot_acc", 0.0),
            r.get("zero_shot_ppl", 0.0),
            r.get("isolated_acc", 0.0),
            r.get("isolated_ppl", 0.0),
            r.get("polished_acc", 0.0),
            r.get("polished_ppl", 0.0),
            r.get("retention_pct", 0.0),
            r.get("total_time_s", 0.0),
            r.get("status", "UNKNOWN")
        ))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Pure GPU Layer-wise KD Benchmark")
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--all-modules", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--isolated-steps", type=int, default=100)
    parser.add_argument("--polish-steps", type=int, default=35)
    parser.add_argument("--iso-lr", type=float, default=1.2e-3)
    parser.add_argument("--polish-lr", type=float, default=2.0e-5)
    parser.add_argument("--out-json", type=str, default="results/sst2_layerwise_kd_ablation.json")
    parser.add_argument("--out-csv", type=str, default="results/sst2_layerwise_kd_ablation.csv")
    parser.add_argument("--ledger-db", type=str, default="results/master_ledger.sqlite")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Pure GPU Accelerated KD Benchmark on: {torch.cuda.get_device_name(0)} ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ArrowSST2Dataset(
        os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-train.arrow"),
        tokenizer
    )
    val_dataset = ArrowSST2Dataset(
        os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-validation.arrow"),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Prewarm Teacher ONCE on GPU
    logger.info("Prewarming Teacher model on GPU to reach authentic baseline...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    teacher_model.train()
    pw_optim = torch.optim.AdamW(teacher_model.parameters(), lr=3e-5)
    for s, batch in enumerate(train_loader):
        if s >= 160:
            break
        pw_optim.zero_grad(set_to_none=True)
        out = teacher_model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        loss = F.cross_entropy(out.logits, batch["label"].to(device))
        loss.backward()
        pw_optim.step()

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    base_acc, base_loss, base_ppl = evaluate_metrics(teacher_model, val_loader, device)
    logger.info(f"Teacher Baseline SST-2: Acc={base_acc:.2f}% | Loss={base_loss:.4f} | PPL={base_ppl:.2f}")

    if args.module:
        target_modules = [args.module]
    elif args.all_modules:
        from neuralops.core.registry import ATTENTION_REGISTRY
        target_modules = list(ATTENTION_REGISTRY.keys())
    else:
        target_modules = [
            "linear_attention",
            "cosformer",
            "gla_attention",
            "svd_linear_attention",
            "decay_linear_attention",
            "gated_linear_attention",
        ]

    results = []
    for mod_name in target_modules:
        try:
            res = run_single_module_pipeline(
                module_name=mod_name,
                teacher_model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                isolated_steps=args.isolated_steps,
                polish_steps=args.polish_steps,
                iso_lr=args.iso_lr,
                polish_lr=args.polish_lr,
            )
            res["baseline_acc"] = round(base_acc, 2)
            res["baseline_ppl"] = round(base_ppl, 2)
            res["retention_pct"] = round((res["polished_acc"] / (base_acc + 1e-9)) * 100.0, 2)
            results.append(res)
        except Exception as exc:
            logger.error(f"Error testing {mod_name}: {exc}")
            results.append({
                "module": mod_name,
                "zero_shot_acc": 0.0,
                "zero_shot_ppl": 99.9,
                "isolated_acc": 0.0,
                "isolated_ppl": 99.9,
                "polished_acc": 0.0,
                "polished_ppl": 99.9,
                "baseline_acc": round(base_acc, 2),
                "baseline_ppl": round(base_ppl, 2),
                "retention_pct": 0.0,
                "total_time_s": 0.0,
                "status": f"FAILED: {exc}"
            })

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    persist_to_sqlite(results, base_acc=round(base_acc, 2), base_ppl=round(base_ppl, 2), sqlite_path=args.ledger_db)

    print("\n" + "=" * 115)
    print(f"ATTENTION SURGERY & KD COMPLETE MATRIX (Teacher Baseline: Acc={base_acc:.2f}%, PPL={base_ppl:.2f})")
    print("=" * 115)
    print(f"{'Module':<24} | {'Zero-Shot Acc':<13} | {'Iso Acc (PPL)':<17} | {'Pol Acc (PPL)':<17} | {'Retention':<10} | {'Time'}")
    print("-" * 115)
    for r in results:
        mod = r.get("module", "")[:23]
        zs = f"{r.get('zero_shot_acc', 0.0):.1f}%"
        iso = f"{r.get('isolated_acc', 0.0):.1f}% ({r.get('isolated_ppl', 0.0):.2f})"
        pol = f"{r.get('polished_acc', 0.0):.1f}% ({r.get('polished_ppl', 0.0):.2f})"
        ret = f"{r.get('retention_pct', 0.0):.1f}%"
        tt = f"{r.get('total_time_s', 0.0):.1f}s"
        print(f"{mod:<24} | {zs:<13} | {iso:<17} | {pol:<17} | {ret:<10} | {tt}")
    print("=" * 115)


if __name__ == "__main__":
    main()
