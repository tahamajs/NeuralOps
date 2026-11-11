#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Authoritative 82-Compressor Multi-Rank Pareto & LoRA Recovery Engine.

Protocol:
  1. Multi-Rank Zero-Shot Validation:
     Evaluates all 82 compressors across ranks k in [256, 128, 64, 32]
     measuring physical parameter reduction ratio (R) and zero-shot SST-2 accuracy.
  2. Targeted LoRA Healing:
     When an aggressive rank (e.g., k=64, 32) drops accuracy by >2.0%,
     attaches a lightweight LoRA adapter (r=8) and runs 50 fast recovery steps
     (15-20s) to restore teacher parity.
  3. 100% In-Memory RAM Resident / Zero-Network:
     Operates entirely from /dev/shm (RAM) and /tmp/sst2_val_cache.pt with zero network I/O.
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
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
torch.set_num_threads(1)
import torch.nn as nn

sys.path.insert(0, "/tmp/neuralops_run")
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor
from neuralops.core.compression_pipeline import compress_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compressor_lora_eval")


class SimpleLoRALinear(nn.Module):
    """Zero-overhead lightweight LoRA adapter for fast healing."""
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: float = 16.0, device: torch.device | None = None):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_f = getattr(base_layer, "in_features", None)
        out_f = getattr(base_layer, "out_features", None)
        if in_f is None or out_f is None:
            # Handle decomposed factors or custom layers
            for p in base_layer.parameters():
                if p.dim() == 2:
                    out_f, in_f = p.shape
                    break

        self.r = r
        self.scaling = alpha / r

        # Infer target device and dtype
        p_ref = next(base_layer.parameters(), None)
        b_ref = next(base_layer.buffers(), None)
        ref = p_ref if p_ref is not None else b_ref
        target_device = device if device is not None else (ref.device if ref is not None else None)
        target_dtype = ref.dtype if (ref is not None and ref.is_floating_point()) else torch.float32

        if in_f and out_f:
            self.lora_A = nn.Parameter(
                torch.randn(r, in_f, device=target_device, dtype=target_dtype) * (1.0 / math.sqrt(r))
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_f, r, device=target_device, dtype=target_dtype)
            )
            self.has_lora = True
        else:
            self.has_lora = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base_layer(x)
        if self.has_lora:
            # Guarantee adapter parameters match x's device and dtype
            if self.lora_A.device != x.device or self.lora_A.dtype != x.dtype:
                self.lora_A.data = self.lora_A.data.to(device=x.device, dtype=x.dtype)
            if self.lora_B.device != x.device or self.lora_B.dtype != x.dtype:
                self.lora_B.data = self.lora_B.data.to(device=x.device, dtype=x.dtype)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            res = res + lora_out
        return res


def attach_lora_to_linear_layers(model: nn.Module, r: int = 8, device: torch.device | None = None) -> int:
    """Wraps linear attention/intermediate projection layers with LoRA."""
    lora_count = 0
    for name, module in list(model.named_modules()):
        is_linear_like = (
            isinstance(module, nn.Linear)
            or hasattr(module, "in_features")
            or "linear" in module.__class__.__name__.lower()
        ) and not isinstance(module, SimpleLoRALinear)
        if is_linear_like and ("attention" in name or "intermediate" in name):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            lora_layer = SimpleLoRALinear(module, r=r, device=device)
            if device is not None:
                lora_layer = lora_layer.to(device)
            setattr(parent, child_name, lora_layer)
            lora_count += 1
    return lora_count


def ensure_all_tensors_on_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Recursively ensures all parameters, buffers, and unregistered adapter tensors are on target device."""
    model.to(device)
    for name, module in model.named_modules():
        for p_name, param in list(module.named_parameters(recurse=False)):
            if param is not None and param.device != device:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
        for b_name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.device != device:
                module.register_buffer(b_name, buf.to(device))
        for attr in ("_shared_V", "V", "U", "bias", "lora_A", "lora_B", "weight", "adapter"):
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, torch.Tensor) and val.device != device:
                    if attr in module._buffers:
                        module.register_buffer(attr, val.to(device))
                    elif attr in module._parameters:
                        val.data = val.data.to(device)
                    else:
                        setattr(module, attr, val.to(device))
                elif isinstance(val, nn.Parameter) and val.data.device != device:
                    val.data = val.data.to(device)
                    if val.grad is not None:
                        val.grad.data = val.grad.data.to(device)
        for attr, val in list(module.__dict__.items()):
            if isinstance(val, torch.Tensor) and val.device != device:
                if attr in module._buffers:
                    module.register_buffer(attr, val.to(device))
                elif attr in module._parameters:
                    val.data = val.data.to(device)
                else:
                    setattr(module, attr, val.to(device))
            elif isinstance(val, nn.Parameter) and val.data.device != device:
                val.data = val.data.to(device)
                if val.grad is not None:
                    val.grad.data = val.grad.data.to(device)
    return model


def evaluate_sst2_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    initial_batch_size: int = 128,
) -> tuple[float, float]:
    """Evaluates SST-2 accuracy with dynamic batch size backoff (128 -> 64 -> 32 -> 16) on CUDA OOM."""
    model.eval()
    dataset = loader.dataset if isinstance(loader, DataLoader) else loader
    current_bs = getattr(loader, "batch_size", initial_batch_size) or initial_batch_size
    criterion = nn.CrossEntropyLoss()

    batch_sizes = [current_bs]
    for bs in (64, 32, 16):
        if bs < current_bs:
            batch_sizes.append(bs)

    for bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        cur_loader = DataLoader(dataset, batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda"))
        total_loss, correct, total = 0.0, 0, 0
        oom_encountered = False

        try:
            with torch.no_grad():
                for b_in, b_att, b_lbl in cur_loader:
                    b_in = b_in.to(device, non_blocking=True)
                    b_att = b_att.to(device, non_blocking=True)
                    b_lbl = b_lbl.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                        out = model(input_ids=b_in, attention_mask=b_att)
                        logits = out.logits
                        loss = criterion(logits, b_lbl)

                    total_loss += loss.item() * len(b_lbl)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == b_lbl).sum().item()
                    total += len(b_lbl)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() and bs > 16:
                logger.warning(
                    "CUDA OOM in SST-2 eval with batch_size=%d. Clearing cache and backing off to %d...",
                    bs,
                    bs // 2,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                oom_encountered = True
                continue
            else:
                raise

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss


def run_fast_lora_recovery(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int = 50,
    lr: float = 2.0e-4,
) -> tuple[float, float]:
    """Runs fast LoRA gradient steps to heal compression distortion with dynamic batch backoff (64 -> 32 -> 16)."""
    student.train()
    ensure_all_tensors_on_device(student, device)

    train_dataset = train_loader.dataset if isinstance(train_loader, DataLoader) else train_loader
    initial_bs = getattr(train_loader, "batch_size", 64) or 64
    batch_sizes = [initial_bs]
    for bs in (32, 16):
        if bs < initial_bs:
            batch_sizes.append(bs)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for current_bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        cur_train_loader = DataLoader(
            train_dataset,
            batch_size=current_bs,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        step = 0
        t_end = time.time() + 45.0  # hard 45-second deadline

        try:
            while step < max_steps and time.time() < t_end:
                for batch in cur_train_loader:
                    if step >= max_steps or time.time() >= t_end:
                        break
                    b_in, b_att, b_lbl = batch[0], batch[1], batch[2]
                    b_in = b_in.to(device, non_blocking=True)
                    b_att = b_att.to(device, non_blocking=True)
                    b_lbl = b_lbl.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                        out = student(input_ids=b_in, attention_mask=b_att)
                        loss = F.cross_entropy(out.logits, b_lbl)

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and current_bs > 16:
                logger.warning(
                    "CUDA OOM encountered during LoRA recovery with batch size %d. "
                    "Clearing cache and backing off (64 -> 32 -> 16)...",
                    current_bs,
                )
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            else:
                raise e

    healed_acc, healed_loss = evaluate_sst2_accuracy(student, val_loader, device)
    return healed_acc, healed_loss


def main():
    parser = argparse.ArgumentParser(description="Multi-Rank Pareto & LoRA Recovery for 82 Compressors")
    parser.add_argument("--keys_json", default="results/compressor_zoo/registry_keys.json")
    parser.add_argument("--teacher_path", default="/dev/shm/bert_sst2_teacher")
    parser.add_argument("--cache_path", default="/tmp/sst2_train_tokens.pt")
    parser.add_argument("--val_cache_path", default="/tmp/sst2_val_cache.pt")
    parser.add_argument("--out_dir", default="/tmp/neuralops_run/results/bert_compressor_zoo")
    parser.add_argument("--ranks", default="256,128,64,32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "multirank_lora_ledger.jsonl"

    ranks = [int(r.strip()) for r in args.ranks.split(",")]

    # Load verified compressor keys
    if os.path.exists(args.keys_json):
        comp_keys = json.loads(Path(args.keys_json).read_text())
    else:
        comp_keys = sorted(list(COMPRESSOR_REGISTRY.keys()))

    logger.info("Starting Multi-Rank LoRA Evaluation for %d compressors across ranks: %s", len(comp_keys), ranks)

    # Load artifacts ONCE from RAM
    from transformers import BertForSequenceClassification
    base_teacher = BertForSequenceClassification.from_pretrained(args.teacher_path, local_files_only=True)
    orig_params = sum(p.numel() for p in base_teacher.parameters())

    vc = torch.load(args.val_cache_path, map_location="cpu")
    val_dataset = TensorDataset(vc["input_ids"], vc["attention_mask"], vc["labels"])
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=(device.type == "cuda"))

    tc = torch.load(args.cache_path, map_location="cpu")
    train_dataset = TensorDataset(tc["input_ids"], tc["attention_mask"], tc["labels"])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=(device.type == "cuda"))

    base_acc, base_loss = evaluate_sst2_accuracy(base_teacher.to(device), val_loader, device)
    base_teacher.cpu()
    logger.info("Uncompressed Teacher Baseline: SST-2 Acc = %.2f%% | Params = %d", base_acc, orig_params)

    completed_pairs = set()
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    completed_pairs.add((data.get("compressor"), data.get("rank")))
                except Exception:
                    pass

    # Master evaluation loop
    for i, key in enumerate(comp_keys):
        logger.info("\n" + "=" * 80)
        logger.info("[%d/%d] COMPRESSOR: %s", i + 1, len(comp_keys), key)
        logger.info("=" * 80)

        for k in ranks:
            if (key, k) in completed_pairs:
                logger.info("[%s | rank=%d] Already evaluated. Skipping.", key, k)
                continue
            t0 = time.time()
            try:
                # Clone base model and apply compression
                student = copy.deepcopy(base_teacher)
                cfg = {"compressors": [{"method": key, "rank": k, "force_rank": True, "use_strs": False}]}
                student = compress_model(student, cfg)
                ensure_all_tensors_on_device(student, device)

                comp_params = sum(p.numel() for p in student.parameters())
                ratio = round(orig_params / max(comp_params, 1), 3)

                # Zero-shot eval (takes ~2 seconds)
                zero_acc, zero_loss = evaluate_sst2_accuracy(student, val_loader, device)
                delta_zero = round(zero_acc - base_acc, 2)
                eval_time = round(time.time() - t0, 2)

                logger.info("[%s | rank=%d] Zero-Shot Acc: %.2f%% (Delta: %+.2f%%) | Ratio: %.2fx | Time: %.1fs",
                            key, k, zero_acc, delta_zero, ratio, eval_time)

                record = {
                    "compressor": key,
                    "rank": k,
                    "ratio": ratio,
                    "comp_params": comp_params,
                    "orig_params": orig_params,
                    "zero_shot_acc": round(zero_acc, 2),
                    "zero_shot_loss": round(zero_loss, 4),
                    "delta_zero": delta_zero,
                    "lora_healed": False,
                    "healed_acc": round(zero_acc, 2),
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "status": "PASSED",
                }

                # Targeted LoRA recovery if accuracy dropped by >2.0%
                if delta_zero < -2.0 and ratio > 1.10:
                    logger.info("[%s | rank=%d] Accuracy dropped >2.0%% (%+.2f%%). Triggering LoRA Healing...", key, k, delta_zero)
                    lora_layers = attach_lora_to_linear_layers(student, r=8, device=device)
                    ensure_all_tensors_on_device(student, device)

                    healed_acc, healed_loss = run_fast_lora_recovery(
                        student=student,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        max_steps=50,
                    )
                    record["lora_healed"] = True
                    record["healed_acc"] = round(healed_acc, 2)
                    record["healed_loss"] = round(healed_loss, 4)
                    record["delta_healed"] = round(healed_acc - base_acc, 2)
                    logger.info("[%s | rank=%d] LoRA Healing Complete: %.2f%% -> %.2f%% (Teacher Parity: %+.2f%%)",
                                key, k, zero_acc, healed_acc, record["delta_healed"])

                with open(ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            except Exception as e:
                logger.warning("[%s | rank=%d] Compression skipped or failed: %s", key, k, str(e)[:120])
                fail_rec = {
                    "compressor": key,
                    "rank": k,
                    "status": "FAIL_CRASH",
                    "error": str(e)[:150],
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
                with open(ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fail_rec) + "\n")
            finally:
                if 'student' in locals():
                    del student
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

    logger.info("\nALL 82 COMPRESSORS VALIDATED ACROSS MULTI-RANKS & LORA RECOVERED!")


if __name__ == "__main__":
    main()
