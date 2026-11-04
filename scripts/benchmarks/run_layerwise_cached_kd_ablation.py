#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajlesi/NeuralOps/graphs/contributors
"""
Unified Offline Teacher Activation Caching & Layer-wise KD Benchmark Engine.

Features:
  1. Offline Teacher Prewarming & Checkpoint Persistence
  2. Persistent Disk Caching of all 12 Attention Layer Inputs & Target Outputs
  3. Direct Cloud Sync to Hugging Face Hub (private dataset repo)
  4. Ultra-fast Zero-Overhead Evaluation of Attention Modules from Disk Cache
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_orig_named_params = nn.Module.named_parameters
def _safe_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _safe_named_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("layerwise_cached_kd")


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


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
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


def build_and_save_offline_cache(
    teacher_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    cache_path: str,
    teacher_ckpt_path: str,
    num_batches: int = 50,
    hf_token: str | None = None,
    hf_repo: str = "tahamajs/neuralops-sst2-layerwise-cache"
) -> dict[str, Any]:
    """Caches all 12 attention layer activations + teacher logits, saves to disk and uploads to HF."""
    logger.info(f"Extracting teacher activations across {num_batches} batches...")
    teacher_model.eval()
    num_layers = len(teacher_model.bert.encoder.layer)
    layer_cache = [{"inputs": [], "masks": [], "targets": []} for _ in range(num_layers)]
    logits_cache = []

    hooks = []
    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            in_t = input[0].detach().cpu()
            mask_t = input[1].detach().cpu() if len(input) > 1 and input[1] is not None else None
            out_t = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
            layer_cache[layer_idx]["inputs"].append(in_t)
            layer_cache[layer_idx]["masks"].append(mask_t)
            layer_cache[layer_idx]["targets"].append(out_t)
        return hook_fn

    for idx, layer in enumerate(teacher_model.bert.encoder.layer):
        hooks.append(layer.attention.self.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for b_idx, batch in enumerate(train_loader):
            if b_idx >= num_batches:
                break
            in_ids = batch["input_ids"].to(device)
            att_m = batch["attention_mask"].to(device)
            out = teacher_model(input_ids=in_ids, attention_mask=att_m)
            logits_cache.append(out.logits.detach().cpu())

    for h in hooks:
        h.remove()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_dict = {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "num_batches": num_batches,
            "num_layers": num_layers,
            "batch_size": train_loader.batch_size,
        },
        "layers": layer_cache,
        "teacher_logits": logits_cache
    }

    # Save cache & checkpoint to disk
    logger.info(f"Saving persistent activation cache to: {cache_path}")
    torch.save(cache_dict, cache_path)
    logger.info(f"Saving prewarmed teacher checkpoint to: {teacher_ckpt_path}")
    torch.save(teacher_model.state_dict(), teacher_ckpt_path)

    # Push to Hugging Face if token provided
    if hf_token:
        try:
            logger.info(f"Uploading cache & teacher checkpoint to Hugging Face: {hf_repo}...")
            import huggingface_hub
            api = huggingface_hub.HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=cache_path,
                path_in_repo=os.path.basename(cache_path),
                repo_id=hf_repo,
                repo_type="dataset"
            )
            api.upload_file(
                path_or_fileobj=teacher_ckpt_path,
                path_in_repo=os.path.basename(teacher_ckpt_path),
                repo_id=hf_repo,
                repo_type="dataset"
            )
            logger.info("Successfully uploaded offline cache and teacher weights to Hugging Face Hub!")
        except Exception as e:
            logger.warning(f"Hugging Face upload failed: {e}")

    return cache_dict


def run_cached_and_polish_ablation(
    module_name: str,
    teacher_state_dict: dict[str, Any],
    layer_caches: list[dict[str, Any]],
    teacher_logits: list[torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    isolated_steps: int = 120,
    polish_steps: int = 40,
    iso_lr: float = 1e-3,
    polish_lr: float = 2e-5,
) -> dict[str, Any]:
    from neuralops.core.registry import create_attention

    logger.info(f"\n{'='*75}\n[3-STAGE PIPELINE] Testing Module: {module_name} (Iso-LR={iso_lr}, Polish-LR={polish_lr})\n{'='*75}")
    t0 = time.time()

    # Stage 0: Zero-shot Surgery from prewarmed teacher weights
    student_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    student_model.load_state_dict(teacher_state_dict)

    for layer in student_model.bert.encoder.layer:
        layer.attention.self = create_attention(module_name, embed_dim=768, num_heads=12)

    student_model.to(device)
    zero_shot_acc, zero_shot_loss = evaluate_accuracy(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 0 (Zero-Shot Surgery): Acc={zero_shot_acc:.2f}% | ValLoss={zero_shot_loss:.4f}")

    # Stage 1: Isolated Attention Mimicry on Cached Activations
    t_iso_start = time.time()
    for l_idx, layer in enumerate(student_model.bert.encoder.layer):
        attn_module = layer.attention.self
        attn_module.train()
        attn_module.to(device)
        optimizer = torch.optim.AdamW(attn_module.parameters(), lr=iso_lr, weight_decay=1e-4)

        inputs = layer_caches[l_idx]["inputs"]
        masks = layer_caches[l_idx]["masks"]
        targets = layer_caches[l_idx]["targets"]
        n_samples = len(inputs)

        for step in range(isolated_steps):
            idx = step % n_samples
            in_t = inputs[idx].to(device)
            mask_t = masks[idx].to(device) if masks[idx] is not None else None
            target_t = targets[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = attn_module(in_t, mask_t) if mask_t is not None else attn_module(in_t)
            out_tensor = out[0] if isinstance(out, tuple) else out

            mse_loss = F.mse_loss(out_tensor, target_t)
            cos_loss = 1.0 - F.cosine_similarity(out_tensor, target_t, dim=-1).mean()
            loss = mse_loss + 0.5 * cos_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(attn_module.parameters(), 1.0)
            optimizer.step()

    t_iso_dur = time.time() - t_iso_start
    isolated_acc, isolated_loss = evaluate_accuracy(student_model, val_loader, device)
    logger.info(f"[{module_name}] Stage 1 (Isolated Mimicry): Acc={isolated_acc:.2f}% | ValLoss={isolated_loss:.4f} | Time={t_iso_dur:.1f}s")

    # Stage 2: Quick Polish E2E
    t_pol_start = time.time()
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=polish_lr, weight_decay=0.01)

    for step_p, batch in enumerate(train_loader):
        if step_p >= polish_steps:
            break
        in_ids = batch["input_ids"].to(device, non_blocking=True)
        att_m = batch["attention_mask"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        s_out = student_model(input_ids=in_ids, attention_mask=att_m)

        if step_p < len(teacher_logits):
            t_logits = teacher_logits[step_p].to(device)
            kd_loss = F.kl_div(
                F.log_softmax(s_out.logits / 2.0, dim=-1),
                F.softmax(t_logits / 2.0, dim=-1),
                reduction="batchmean"
            ) * 4.0
        else:
            kd_loss = 0.0

        ce_loss = F.cross_entropy(s_out.logits, lbls)
        total_l = (0.5 * kd_loss + 0.5 * ce_loss) if isinstance(kd_loss, torch.Tensor) else ce_loss
        total_l.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()

    t_pol_dur = time.time() - t_pol_start
    final_acc, final_loss = evaluate_accuracy(student_model, val_loader, device)
    total_dur = time.time() - t0
    logger.info(f"[{module_name}] Stage 2 (Quick Polish): Acc={final_acc:.2f}% | ValLoss={final_loss:.4f} | TotalTime={total_dur:.1f}s")

    del student_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "module": module_name,
        "zero_shot_acc": round(zero_shot_acc, 2),
        "isolated_mimicry_acc": round(isolated_acc, 2),
        "polished_e2e_acc": round(final_acc, 2),
        "zero_shot_loss": round(zero_shot_loss, 4),
        "isolated_loss": round(isolated_loss, 4),
        "polished_loss": round(final_loss, 4),
        "isolated_time_s": round(t_iso_dur, 1),
        "polish_time_s": round(t_pol_dur, 1),
        "total_time_s": round(total_dur, 1),
        "status": "PASSED"
    }


def persist_to_sqlite(results: list[dict[str, Any]], sqlite_path: str = "results/master_ledger.sqlite") -> None:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS layerwise_kd_ablation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            module TEXT,
            zero_shot_acc REAL,
            isolated_acc REAL,
            polished_acc REAL,
            retention_pct REAL,
            total_time_s REAL,
            status TEXT
        )
    """)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for r in results:
        cur.execute("""
            INSERT INTO layerwise_kd_ablation (
                timestamp, module, zero_shot_acc, isolated_acc, polished_acc,
                retention_pct, total_time_s, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            r.get("module"),
            r.get("zero_shot_acc", 0.0),
            r.get("isolated_mimicry_acc", 0.0),
            r.get("polished_e2e_acc", 0.0),
            r.get("retention_pct", 0.0),
            r.get("total_time_s", 0.0),
            r.get("status", "UNKNOWN")
        ))
    conn.commit()
    conn.close()
    logger.info(f"Persisted {len(results)} records to master ledger SQLite: {sqlite_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Offline Cached Layer-wise KD Benchmark")
    parser.add_argument("--module", type=str, default=None, help="Specific module to run (e.g. linear_attention)")
    parser.add_argument("--all-modules", action="store_true", help="Run over all eligible attention modules")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-batches", type=int, default=40)
    parser.add_argument("--isolated-steps", type=int, default=120)
    parser.add_argument("--polish-steps", type=int, default=40)
    parser.add_argument("--iso-lr", type=float, default=8e-4, help="Learning rate for isolated attention mimicry")
    parser.add_argument("--polish-lr", type=float, default=1.5e-5, help="Learning rate for quick polish")
    parser.add_argument("--cache-file", type=str, default="results/teacher_layerwise_cache_sst2.pt")
    parser.add_argument("--teacher-ckpt", type=str, default="results/teacher_sst2_checkpoint.pt")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf-repo", type=str, default="tahamajs/neuralops-sst2-layerwise-cache")
    parser.add_argument("--force-recache", action="store_true", help="Force rebuilding cache even if exists")
    parser.add_argument("--out-json", type=str, default="results/sst2_cached_layerwise_kd.json")
    parser.add_argument("--out-csv", type=str, default="results/sst2_cached_layerwise_kd.csv")
    parser.add_argument("--ledger-db", type=str, default="results/master_ledger.sqlite")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Launching Offline Cached KD Engine on: {torch.cuda.get_device_name(0)}")

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

    # 1. Load or Generate Offline Cache
    if os.path.exists(args.cache_file) and os.path.exists(args.teacher_ckpt) and not args.force_recache:
        logger.info(f"Loading persistent cache from: {args.cache_file}")
        cache_data = torch.load(args.cache_file, map_location="cpu")
        teacher_state_dict = torch.load(args.teacher_ckpt, map_location="cpu")
        layer_caches = cache_data["layers"]
        teacher_logits = cache_data["teacher_logits"]

        temp_teacher = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
        temp_teacher.load_state_dict(teacher_state_dict)
        base_acc, _ = evaluate_accuracy(temp_teacher, val_loader, device)
        del temp_teacher
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("Building fresh persistent cache from prewarmed teacher...")
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

        base_acc, _ = evaluate_accuracy(teacher_model, val_loader, device)
        logger.info(f"Teacher Baseline Accuracy on SST-2: {base_acc:.2f}%")

        cache_data = build_and_save_offline_cache(
            teacher_model=teacher_model,
            train_loader=train_loader,
            device=device,
            cache_path=args.cache_file,
            teacher_ckpt_path=args.teacher_ckpt,
            num_batches=args.cache_batches,
            hf_token=args.hf_token,
            hf_repo=args.hf_repo,
        )
        teacher_state_dict = torch.load(args.teacher_ckpt, map_location="cpu")
        layer_caches = cache_data["layers"]
        teacher_logits = cache_data["teacher_logits"]
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Verified Teacher Baseline: {base_acc:.2f}% (Cache ready)")

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
            res = run_cached_and_polish_ablation(
                module_name=mod_name,
                teacher_state_dict=teacher_state_dict,
                layer_caches=layer_caches,
                teacher_logits=teacher_logits,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                isolated_steps=args.isolated_steps,
                polish_steps=args.polish_steps,
                iso_lr=args.iso_lr,
                polish_lr=args.polish_lr,
            )
            res["baseline_acc"] = round(base_acc, 2)
            res["retention_pct"] = round((res["polished_e2e_acc"] / (base_acc + 1e-9)) * 100.0, 2)
            results.append(res)
        except Exception as exc:
            logger.error(f"Error testing {mod_name}: {exc}")
            results.append({
                "module": mod_name,
                "zero_shot_acc": 0.0,
                "isolated_mimicry_acc": 0.0,
                "polished_e2e_acc": 0.0,
                "baseline_acc": round(base_acc, 2),
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
    persist_to_sqlite(results, sqlite_path=args.ledger_db)

    print("\n" + "=" * 105)
    print(f"OFFLINE CACHED LAYER-WISE KD SUMMARY ON SST-2 (Baseline: {base_acc:.2f}%)")
    print("=" * 105)
    print(f"{'Module':<25} | {'Zero-Shot':<12} | {'Isolated':<14} | {'Polished':<14} | {'Retention':<12} | {'Time':<10}")
    print("-" * 105)
    for r in results:
        mod = r.get("module", "")[:24]
        zs = f"{r.get('zero_shot_acc', 0.0):.2f}%"
        iso = f"{r.get('isolated_mimicry_acc', 0.0):.2f}%"
        pol = f"{r.get('polished_e2e_acc', 0.0):.2f}%"
        ret = f"{r.get('retention_pct', 0.0):.2f}%"
        tt = f"{r.get('total_time_s', 0.0):.1f}s"
        print(f"{mod:<25} | {zs:<12} | {iso:<14} | {pol:<14} | {ret:<12} | {tt:<10}")
    print("=" * 105)


if __name__ == "__main__":
    main()
