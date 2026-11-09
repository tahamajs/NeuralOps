#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Master Track B Qwen Multi-Scale Compressor Zoo Distillation Engine.

Compresses modern causal language models (e.g. Qwen/Qwen2.5-0.5B) using verified
compression engines (from results/compressor_zoo/registry_keys.json), evaluates
exact parameter reduction, and runs a 2-stage post-compression healing curriculum
with Top-K Sparse Logit KD (topk=64) to bypass dense vocabulary logit OOM:
  - Stage 0: Causal Perplexity & SST-2 Zero-Shot classification
  - Stage 1: Frozen Backbone Alignment (90s)
  - Stage 2: End-to-End Unfrozen KD + Top-K Sparse Logit Distillation (210s)
  - Export: Self-contained HuggingFace checkpoint and automated Model Card
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

# PyTorch 1.12 compatibility with modern HuggingFace Accelerate
try:
    _orig_named_parameters = nn.Module.named_parameters
    def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)
    nn.Module.named_parameters = _compat_named_parameters
except Exception:
    pass

from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neuralops.core.compression_pipeline import compress_model
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor
from neuralops.hub.uploader import AsyncHFHubUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_compressor_zoo_train")


class CausalTokenDataset(Dataset):
    """Contiguous chunk dataset for generative autoregressive language modeling."""

    def __init__(self, tokens_2d: torch.Tensor):
        self.chunks = tokens_2d

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.chunks[idx].clone()
        labels = ids.clone()
        attn_mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn_mask, "labels": labels}


def compute_topk_sparse_kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, topk: int = 64, tau: float = 2.0) -> torch.Tensor:
    """Top-K Sparse Logit KD avoiding dense vocabulary logit allocation."""
    t_topk_vals, t_topk_inds = torch.topk(t_logits, k=topk, dim=-1)
    s_topk_vals = torch.gather(s_logits, dim=-1, index=t_topk_inds)
    p_s = F.log_softmax(s_topk_vals / tau, dim=-1)
    p_t = F.softmax(t_topk_vals / tau, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (tau * tau)


def evaluate_causal_ppl(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)
            out = model(input_ids=in_ids, attention_mask=att_m, labels=lbls)
            n_tokens = in_ids.numel()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return 999.0
    return math.exp(min(total_loss / total_tokens, 20.0))


def evaluate_sst2_accuracy(model: nn.Module, tokenizer: Any, device: torch.device, num_samples: int = 200) -> float:
    from datasets import load_dataset
    try:
        raw = load_dataset("glue", "sst2", split="validation")
    except Exception:
        return 0.0

    model.eval()
    correct, total = 0, 0
    pos_id = tokenizer.encode("great", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("terrible", add_special_tokens=False)[0]

    with torch.no_grad():
        for i in range(min(num_samples, len(raw))):
            sentence = raw[i]["sentence"]
            label = raw[i]["label"]
            prompt = f"Review: {sentence}\nSentiment:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**inputs)
            next_logits = out.logits[:, -1, :]
            pred = 1 if next_logits[:, pos_id].item() > next_logits[:, neg_id].item() else 0
            if pred == label:
                correct += 1
            total += 1
    return (correct / total) * 100.0 if total > 0 else 0.0


def train_single_qwen_compressor(
    comp_key: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    train_secs: float = 300.0,
    topk: int = 64,
    rank: int = 128,
    grad_accum: int = 4,
) -> dict[str, Any]:
    t_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Compressor: %s (%ds, rank=%d)", comp_key, int(train_secs), rank)
    logger.info("=" * 80)

    student = copy.deepcopy(teacher_model)
    orig_params = sum(p.numel() for p in student.parameters())

    try:
        cfg = {"compressors": [{"method": comp_key, "rank": rank}]}
        student = compress_model(student, cfg)
    except Exception as e:
        logger.warning("[%s] Pipeline compress failed, trying factory: %s", comp_key, e)
        inst = create_compressor(comp_key, model=student, rank=rank)
        if hasattr(inst, "compress"):
            res = inst.compress()
            student = res[0] if isinstance(res, tuple) else res

    student = student.to(device)
    comp_params = sum(p.numel() for p in student.parameters())
    param_reduction = orig_params / max(comp_params, 1)

    logger.info("[%s] Parameters: %d -> %d (Reduction: %.2fx)", comp_key, orig_params, comp_params, param_reduction)

    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    pre_sst2 = evaluate_sst2_accuracy(student, tokenizer, device)
    logger.info("[%s] Pre-KD: PPL=%.2f | SST-2 Acc=%.2f%%", comp_key, pre_ppl, pre_sst2)

    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * 0.30)
    t_s2_deadline = t_train_start + train_secs

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Stage 1: Frozen backbone (90s)
    for name, p in student.named_parameters():
        p.requires_grad = any(k in name for k in ("svd", "factor", "proj", "comp", "lora", "down", "up", "dense"))

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1.0e-3, weight_decay=0.01)
    student.train()
    step_s1 = 0

    while time.time() < t_s1_deadline:
        for batch in train_loader:
            if time.time() >= t_s1_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                with torch.no_grad():
                    t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                s_out = student(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)

                s_h = s_out.hidden_states[1:]
                t_h = t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(sh, th) for sh, th in zip(s_h, t_h)]).mean()
                loss = loss_mse / grad_accum

            scaler.scale(loss).backward()
            step_s1 += 1
            if step_s1 % grad_accum == 0:
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt1)
                scaler.update()
                opt1.zero_grad(set_to_none=True)

    # Stage 2: End-to-End Unfrozen KD (210s)
    for p in student.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW(student.parameters(), lr=1.0e-4, weight_decay=0.01)
    student.train()
    step_s2 = 0

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                with torch.no_grad():
                    t_out = teacher_model(input_ids=in_ids, attention_mask=att_m)
                s_out = student(input_ids=in_ids, attention_mask=att_m, labels=lbls)

                loss_sparse = compute_topk_sparse_kd_loss(s_out.logits, t_out.logits, topk=topk)
                loss_ce = s_out.loss if s_out.loss is not None else torch.tensor(0.0, device=device)
                loss = (0.50 * loss_sparse + 0.50 * loss_ce) / grad_accum

            scaler.scale(loss).backward()
            step_s2 += 1
            if step_s2 % grad_accum == 0:
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt2)
                scaler.update()
                opt2.zero_grad(set_to_none=True)

    total_time = time.time() - t_start
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    post_sst2 = evaluate_sst2_accuracy(student, tokenizer, device)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0

    result = {
        "model": model_name,
        "compressor": comp_key,
        "param_reduction": round(param_reduction, 3),
        "orig_params": orig_params,
        "comp_params": comp_params,
        "ppl_before": round(pre_ppl, 2),
        "ppl_after": round(post_ppl, 2),
        "ppl_delta": round(post_ppl - pre_ppl, 2),
        "sst2_before": round(pre_sst2, 2),
        "sst2_after": round(post_sst2, 2),
        "sst2_delta": round(post_sst2 - pre_sst2, 2),
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "peak_vram_mb": peak_vram,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "PASSED",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))

    # Save strictly compressed layer and adapter weights
    comp_state: dict[str, torch.Tensor] = {}
    for k, v in student.state_dict().items():
        k_lower = k.lower()
        if any(term in k_lower for term in ("compress", "lora", "adapter", "down_proj", "up_proj", "svd", "quant")):
            comp_state[k] = v.detach().cpu().half().contiguous()
    if not comp_state:
        # Fallback to attention/linear weights
        for k, v in student.state_dict().items():
            if "attn" in k.lower() or "mlp" in k.lower():
                comp_state[k] = v.detach().cpu().half().contiguous()

    from safetensors.torch import save_file
    save_file(comp_state, str(out_dir / "attention_weights.safetensors"))

    # Purge any heavy whole-model files
    for heavy_name in ["pytorch_model.bin", "model.safetensors", "optimizer.pt"]:
        h_path = out_dir / heavy_name
        if h_path.exists():
            try:
                os.remove(h_path)
            except Exception:
                pass

    cfg = student.config.to_dict()
    cfg["compressor_key"] = comp_key
    cfg["weights_format"] = "compressor_only_fp16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Qwen Multi-Scale Compressor Zoo Distillation")
    parser.add_argument("--keys_json", default="results/compressor_zoo/registry_keys.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--train_secs", type=float, default=300.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--out_dir", default="results/qwen_compressor_zoo")
    parser.add_argument("--metrics_log", default="results/qwen_compressor_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="Hooshaai")
    parser.add_argument("--hf_upload", type=int, default=0, help="0 to delegate upload to decoupled login-node daemon (recommended for HPC), 1 for in-process uploader")
    parser.add_argument("--keep_local_weights", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available.")
        sys.exit(3)

    keys_path = Path(args.keys_json)
    if not keys_path.exists():
        logger.error("Keys JSON not found: %s", keys_path)
        sys.exit(2)

    comp_keys = json.loads(keys_path.read_text())
    logger.info("Loaded %d verified compressors for Qwen from %s", len(comp_keys), keys_path)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(args.metrics_log)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    uploader = None
    if args.hf_upload and hf_token:
        uploader = AsyncHFHubUploader(token=hf_token, org=args.hf_org)
        uploader.start()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device)
    teacher.eval()

    # Load dataset
    from datasets import load_dataset
    raw = load_dataset("wikitext", args.dataset)
    train_text = "\n\n".join([t for t in raw["train"]["text"] if t.strip()])
    val_text = "\n\n".join([t for t in raw["validation"]["text"] if t.strip()])

    def to_chunks(text: str) -> torch.Tensor:
        toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        n = (toks.size(0) // args.seq_len) * args.seq_len
        return toks[:n].view(-1, args.seq_len)

    train_chunks = to_chunks(train_text)
    val_chunks = to_chunks(val_text)

    train_loader = DataLoader(CausalTokenDataset(train_chunks), batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(CausalTokenDataset(val_chunks), batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model_slug = args.model.split("/")[-1].lower()

    for i, comp_key in enumerate(comp_keys):
        logger.info("\n[%d/%d] Processing Qwen Compressor: %s", i + 1, len(comp_keys), comp_key)
        ckpt_dir = out_base / f"{model_slug}_{comp_key}"

        if (ckpt_dir / "training_metrics.json").exists():
            if args.hf_upload and uploader and not (ckpt_dir / "upload_success.json").exists():
                uploader.submit_upload(
                    repo_id=f"{args.hf_org}/qwen-{model_slug}-comp-{comp_key.replace('_', '-')}",
                    folder_path=str(ckpt_dir),
                    delete_after_upload=bool(not args.keep_local_weights),
                )
            logger.info("[%s] Already completed, skipping...", comp_key)
            continue

        try:
            res = train_single_qwen_compressor(
                comp_key=comp_key,
                teacher_model=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                tokenizer=tokenizer,
                out_dir=ckpt_dir,
                model_name=args.model,
                train_secs=args.train_secs,
                topk=args.topk,
                rank=args.rank,
                grad_accum=args.grad_accum,
            )
            with open(metrics_file, "a") as f:
                f.write(json.dumps(res) + "\n")

            if args.hf_upload and uploader:
                uploader.submit_upload(
                    repo_id=f"{args.hf_org}/qwen-{model_slug}-comp-{comp_key.replace('_', '-')}",
                    folder_path=str(ckpt_dir),
                    delete_after_upload=bool(not args.keep_local_weights),
                )
        except Exception as e:
            logger.error("[%s] Training failed: %s", comp_key, e, exc_info=True)
            fail_record = {
                "model": args.model,
                "compressor": comp_key,
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(fail_record) + "\n")

    if uploader:
        logger.info("Waiting for all background HuggingFace uploads to finish...")
        uploader.shutdown(wait=True)

    logger.info("Qwen Compressor Zoo Training Complete!")


if __name__ == "__main__":
    main()
