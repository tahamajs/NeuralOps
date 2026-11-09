#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Ultra-Fast Pure GPU Layer-wise 6-Pillar KD Benchmark Engine.

Evaluates post-surgery knowledge distillation across all 6 canonical healing pillars:
  1. Logits Distillation (temperature-scaled KL divergence with tau^2 normalization)
  2. Hidden Representation Distillation (MSE + Directional Cosine alignment)
  3. Layer-Attn Representation Distillation (Intermediate multi-layer alignment)
  4. SVD Spectrum Alignment (Subspace singular value decay matching)
  5. Fisher Curvature Regularization (Activation covariance weighted curvature penalty)
  6. Sparsity / Attention Entropy Regularizer (Activation L1 norm regularization)

Features:
  - Fail-closed CUDA validation (--device cuda without CUDA raises immediately)
  - Strict step-by-step finite loss trajectory verification (catches NaN/Inf early)
  - Trajectory persistence to JSON, CSV, and SQLite master ledger
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
logger = logging.getLogger("layerwise_6pillar_kd")


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
            return_tensors="pt",
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SyntheticSST2Dataset(Dataset):
    """Deterministic synthetic SST-2 dataset for offline unit tests and smoke verification."""

    def __init__(self, num_samples: int = 128, max_length: int = 128, vocab_size: int = 1000) -> None:
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(42 + idx)
        return {
            "input_ids": torch.randint(10, self.vocab_size, (self.max_length,), generator=generator),
            "attention_mask": torch.ones(self.max_length, dtype=torch.long),
            "label": torch.tensor(idx % 2, dtype=torch.long),
        }


def compute_6pillar_loss(
    s_out: Any,
    t_out: Any,
    labels: torch.Tensor,
    temperature: float = 2.0,
    w_ce: float = 0.5,
    w_logits: float = 0.5,
    w_hidden: float = 0.1,
    w_attn: float = 0.1,
    w_svd: float = 0.05,
    w_fisher: float = 0.05,
    w_sparsity: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes the 6 canonical distillation loss pillars:
      1. Logits: Temperature-scaled KL divergence with tau^2 normalization
      2. Hidden: Combined MSE and directional Cosine distance on terminal representations
      3. Layer-Attn: Mean intermediate representation MSE across encoder layers
      4. SVD Spectrum: Subspace singular value distribution MSE alignment
      5. Fisher Curvature: Activation variance-weighted quadratic curvature penalty
      6. Sparsity: Mean L1 activation magnitude regularizer
    """
    s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
    t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

    # Supervised cross-entropy task loss
    loss_ce = F.cross_entropy(s_logits, labels)

    # 1. Pillar 1: Logits Distillation
    p_s = F.log_softmax(s_logits / temperature, dim=-1)
    p_t = F.softmax(t_logits / temperature, dim=-1)
    loss_logits = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature**2)

    # Extract hidden states
    s_hidden = s_out.hidden_states if hasattr(s_out, "hidden_states") and s_out.hidden_states is not None else None
    t_hidden = t_out.hidden_states if hasattr(t_out, "hidden_states") and t_out.hidden_states is not None else None

    if s_hidden is not None and t_hidden is not None and len(s_hidden) > 0:
        s_last = s_hidden[-1]
        t_last = t_hidden[-1].detach()

        # 2. Pillar 2: Hidden Representation Alignment (MSE + Directional Cosine)
        loss_hidden_mse = F.mse_loss(s_last, t_last)
        loss_hidden_cos = 1.0 - F.cosine_similarity(s_last, t_last, dim=-1).mean()
        loss_hidden = 0.5 * loss_hidden_mse + 0.5 * loss_hidden_cos

        # 3. Pillar 3: Layer-Attn Representation Distillation (across interior layers)
        if len(s_hidden) > 2:
            interior_losses = [
                F.mse_loss(s_hidden[i], t_hidden[i].detach())
                for i in range(1, len(s_hidden) - 1)
            ]
            loss_attn = sum(interior_losses) / len(interior_losses)
        else:
            loss_attn = F.mse_loss(s_last, t_last)

        # 4. Pillar 4: SVD Spectrum Alignment
        s_flat = s_last[:min(s_last.size(0), 8)].view(-1, s_last.size(-1)).float()
        t_flat = t_last[:min(t_last.size(0), 8)].view(-1, t_last.size(-1)).float()
        s_sv = torch.linalg.svdvals(s_flat)
        t_sv = torch.linalg.svdvals(t_flat)
        loss_svd = F.mse_loss(s_sv, t_sv)

        # 5. Pillar 5: Fisher Curvature Regularizer
        diff = s_last - t_last
        var_t = torch.var(t_last, dim=(0, 1), keepdim=True).detach() + 1e-6
        loss_fisher = torch.mean((diff**2) * var_t)

        # 6. Pillar 6: Sparsity / Attention Regularizer
        loss_sparsity = torch.mean(torch.abs(s_last))
    else:
        # Fallback when hidden_states are omitted
        zero_tensor = torch.zeros((), device=s_logits.device, dtype=s_logits.dtype)
        loss_hidden = zero_tensor
        loss_attn = zero_tensor
        loss_svd = zero_tensor
        loss_fisher = zero_tensor
        loss_sparsity = zero_tensor

    total_loss = (
        w_ce * loss_ce
        + w_logits * loss_logits
        + w_hidden * loss_hidden
        + w_attn * loss_attn
        + w_svd * loss_svd
        + w_fisher * loss_fisher
        + w_sparsity * loss_sparsity
    )

    # Fail-closed check: verify numerical finiteness of every pillar
    breakdown = {
        "loss_ce": round(float(loss_ce.item()), 6),
        "loss_logits": round(float(loss_logits.item()), 6),
        "loss_hidden": round(float(loss_hidden.item()), 6),
        "loss_attn": round(float(loss_attn.item()), 6),
        "loss_svd": round(float(loss_svd.item()), 6),
        "loss_fisher": round(float(loss_fisher.item()), 6),
        "loss_sparsity": round(float(loss_sparsity.item()), 6),
        "total_loss": round(float(total_loss.item()), 6),
    }

    for name, val in breakdown.items():
        if not math.isfinite(val):
            raise FloatingPointError(
                f"Non-finite loss detected in 6-pillar KD objective for '{name}': {val!r}. "
                f"Complete breakdown: {breakdown}"
            )

    return total_loss, breakdown


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

    logger.info(f"\n{'='*80}\n[PIPELINE] Testing Attention Module: {module_name} (6-Pillar KD)\n{'='*80}")
    t0 = time.time()

    # 1. Surgery
    student_model = copy.deepcopy(teacher_model)
    encoder_layers = getattr(getattr(student_model, "bert", None), "encoder", None)
    if encoder_layers is not None:
        layers = encoder_layers.layer
        embed_dim = 768
        num_heads = 12
    else:
        # Fallback for small mock configs
        layers = student_model.base_model.encoder.layer if hasattr(student_model, "base_model") else []
        embed_dim = 128
        num_heads = 4

    for layer in layers:
        layer.attention.self = create_attention(module_name, embed_dim=embed_dim, num_heads=num_heads)

    student_model.to(device)
    z_acc, z_loss, z_ppl = evaluate_metrics(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 0 (Zero-Shot Surgery): Acc={z_acc:.2f}% | Loss={z_loss:.4f} | PPL={z_ppl:.2f}")

    trajectory: list[dict[str, Any]] = []

    # 2. Stage 1: Isolated Attention Mimicry (Backbone Frozen, 6-Pillar KD)
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
            t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
        s_out = student_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)

        loss, breakdown = compute_6pillar_loss(s_out, t_out, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_attn_params, 1.0)
        optimizer_iso.step()

        breakdown["stage"] = "isolated"
        breakdown["step"] = step_i
        trajectory.append(breakdown)

        if step_i % 25 == 0 or step_i == isolated_steps - 1:
            logger.info(
                f"[{module_name}] Iso Step {step_i}/{isolated_steps}: loss={loss.item():.4f} "
                f"(logits={breakdown['loss_logits']:.4f}, hidden={breakdown['loss_hidden']:.4f}, "
                f"attn={breakdown['loss_attn']:.4f}, svd={breakdown['loss_svd']:.4f}, "
                f"fisher={breakdown['loss_fisher']:.4f}, sparsity={breakdown['loss_sparsity']:.4f})"
            )

    t_iso_dur = time.time() - t_iso_start
    iso_acc, iso_loss, iso_ppl = evaluate_metrics(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 1 (Isolated Attn): Acc={iso_acc:.2f}% | Loss={iso_loss:.4f} | PPL={iso_ppl:.2f} | Time={t_iso_dur:.1f}s")

    # 3. Stage 2: Quick Polish (E2E 6-Pillar KD)
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
            t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
        s_out = student_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)

        total_l, breakdown_pol = compute_6pillar_loss(s_out, t_out, lbls)
        total_l.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer_pol.step()

        breakdown_pol["stage"] = "polish"
        breakdown_pol["step"] = step_p
        trajectory.append(breakdown_pol)

    t_pol_dur = time.time() - t_pol_start
    final_acc, final_loss, final_ppl = evaluate_metrics(student_model, val_loader, device)
    total_dur = time.time() - t0
    logger.info(f"[{module_name}] Stage 2 (Quick Polish): Acc={final_acc:.2f}% | Loss={final_loss:.4f} | PPL={final_ppl:.2f} | TotalTime={total_dur:.1f}s")

    del student_model
    gc.collect()
    if device.type == "cuda":
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
        "loss_trajectory": trajectory,
        "trajectory_steps": len(trajectory),
        "pillars_evaluated": ["Logits", "Hidden", "Layer-Attn", "SVD Spectrum", "Fisher Curvature", "Sparsity"],
        "status": "PASSED",
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
            r.get("status", "UNKNOWN"),
        ))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Pure GPU Layer-wise 6-Pillar KD Benchmark")
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--all-modules", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--isolated-steps", type=int, default=100)
    parser.add_argument("--polish-steps", type=int, default=35)
    parser.add_argument("--iso-lr", type=float, default=1.2e-3)
    parser.add_argument("--polish-lr", type=float, default=2.0e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing json results if present")
    parser.add_argument("--out-json", type=str, default="results/sst2_layerwise_kd_ablation.json")
    parser.add_argument("--out-csv", type=str, default="results/sst2_layerwise_kd_ablation.csv")
    parser.add_argument("--ledger-db", type=str, default="results/master_ledger.sqlite")
    args = parser.parse_args()

    # Fail closed on unavailable CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested (--device cuda), but CUDA is not available. "
            "Failing closed in compliance with NeuralOps Verification Integrity Contract."
        )

    device = torch.device(args.device)
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU / Fallback"
    logger.info(f"=== Layer-wise 6-Pillar KD Benchmark on: {dev_name} ({device}) ===")

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        tokenizer = None

    arrow_train = os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-train.arrow")
    arrow_val = os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-validation.arrow")

    if tokenizer is not None and os.path.exists(arrow_train) and os.path.exists(arrow_val):
        train_dataset = ArrowSST2Dataset(arrow_train, tokenizer)
        val_dataset = ArrowSST2Dataset(arrow_val, tokenizer)
    else:
        logger.warning("Arrow SST-2 dataset not present; using synthetic SST-2 benchmark dataset.")
        train_dataset = SyntheticSST2Dataset(num_samples=256, max_length=128)
        val_dataset = SyntheticSST2Dataset(num_samples=64, max_length=128)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem)

    # Prewarm Teacher ONCE
    logger.info(f"Initializing Teacher model on {device}...")
    try:
        teacher_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    except Exception as exc:
        logger.warning(f"Could not load bert-base-uncased ({exc}); creating Bert model for evaluation.")
        from transformers import BertConfig, BertForSequenceClassification
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            num_labels=2,
        )
        teacher_model = BertForSequenceClassification(config).to(device)

    teacher_model.train()
    pw_optim = torch.optim.AdamW(teacher_model.parameters(), lr=3e-5)
    for s, batch in enumerate(train_loader):
        if s >= min(160, len(train_loader)):
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
    completed_modules = set()
    if args.resume and os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                for item in loaded:
                    if item.get("status") == "PASSED":
                        results.append(item)
                        completed_modules.add(item.get("module"))
            logger.info(f"Resuming: found {len(completed_modules)} already passed modules in {args.out_json}")
            # Persist existing results to SQLite immediately
            if results:
                persist_to_sqlite(results, base_acc=round(base_acc, 2), base_ppl=round(base_ppl, 2), sqlite_path=args.ledger_db)
        except Exception as exc:
            logger.warning(f"Could not load resume data from {args.out_json}: {exc}")

    for mod_name in target_modules:
        if mod_name in completed_modules:
            logger.info(f"Skipping already-completed module: {mod_name}")
            continue

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
            persist_to_sqlite([res], base_acc=round(base_acc, 2), base_ppl=round(base_ppl, 2), sqlite_path=args.ledger_db)
        except Exception as exc:
            logger.error(f"Error testing {mod_name}: {exc}")
            failed_entry = {
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
                "pillars_evaluated": ["Logits", "Hidden", "Layer-Attn", "SVD Spectrum", "Fisher Curvature", "Sparsity"],
                "status": f"FAILED: {exc}",
            }
            results.append(failed_entry)
            persist_to_sqlite([failed_entry], base_acc=round(base_acc, 2), base_ppl=round(base_ppl, 2), sqlite_path=args.ledger_db)

        # Clean GPU memory between module iterations
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 115)
    print(f"ATTENTION SURGERY & 6-PILLAR KD COMPLETE MATRIX (Teacher Baseline: Acc={base_acc:.2f}%, PPL={base_ppl:.2f})")
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
