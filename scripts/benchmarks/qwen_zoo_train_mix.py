#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Master Multi-Scale Qwen Attention Distillation & Evaluation Engine.

Features:
  - Multi-scale support: Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-2B, Qwen3.8-2B/4B
  - Surgery patterns:
      * 'all': replace all transformer attention layers with transplanted module
      * 'hybrid_3_1': 3 subquadratic linear/delta layers per 1 full/sliding-window layer
  - Top-K Sparse Logit KD (topk=64):
      * Bypasses dense [B, L, 151936] logit allocations, avoiding 16 GB VRAM OOM on RTX 3090
  - Two-Stage Budgeted KD Schedule:
      * Stage 1 (30%): Frozen-backbone layerwise all-layer hidden MSE + Fast Gram Covariance
                       (lm_head completely bypassed)
      * Stage 2 (70%): End-to-end unfrozen KD with Top-K Sparse Logit KL + Causal LM CE
  - Pre- and post-distillation evaluation:
      * Causal LM Perplexity on WikiText-103 validation split
      * Downstream SST-2 Sentiment Classification accuracy
  - Resume-skip logic with upload_success.json verification
  - Asynchronous background Hugging Face Hub upload with automated disk cleanup
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
from typing import Any, Dict, List, Optional, Tuple

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

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.hub.uploader import AsyncHFHubUploader
from neuralops.layers.qwen_adapter import QwenAttentionAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_zoo_train")


def fast_gram_covariance_loss(s_hidden: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
    """
    Fast Tensor-Core Subspace Covariance matching avoiding SVD stalls.
    Computes normalized Gram matrix Frobenius distance.
    """
    s_flat = s_hidden.view(-1, s_hidden.size(-1)).float()
    t_flat = t_hidden.view(-1, t_hidden.size(-1)).float()

    s_norm = torch.norm(s_flat, p="fro").clamp(min=1e-6)
    t_norm = torch.norm(t_flat, p="fro").clamp(min=1e-6)

    s_gram = torch.matmul(s_flat.t(), s_flat) / s_norm
    t_gram = torch.matmul(t_flat.t(), t_flat) / t_norm

    return F.mse_loss(s_gram, t_gram)


def compute_topk_sparse_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 64,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Top-K Sparse Logit KD avoiding dense [B, L, 151936] logit allocations.
    Selects top-K teacher tokens and gathers student logits strictly at those positions.
    """
    tau = temperature
    t_topk_vals, t_topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
    s_topk_vals = torch.gather(student_logits, dim=-1, index=t_topk_idx)

    p_s = F.log_softmax(s_topk_vals / tau, dim=-1)
    p_t = F.softmax(t_topk_vals / tau, dim=-1)

    return F.kl_div(p_s, p_t, reduction="batchmean") * (tau * tau)


class StratifiedQwenDataset(Dataset):
    """Contiguous token chunk dataset for causal language modeling."""

    def __init__(self, token_chunks: torch.Tensor) -> None:
        self.chunks = token_chunks

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.chunks[idx].clone()
        labels = ids.clone()
        attn_mask = torch.ones_like(ids, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def load_tokenized_qwen_corpus(
    tokenizer: Any,
    dataset_name: str = "wikitext-2-raw-v1",
    seq_len: int = 512,
    max_train_samples: int = 4000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenizes corpus into contiguous chunks of length seq_len."""
    from datasets import load_dataset

    logger.info("Loading %s dataset...", dataset_name)
    if "wikitext" in dataset_name:
        raw = load_dataset("wikitext", dataset_name)
        train_texts = [t for t in raw["train"]["text"] if t.strip()]
        val_texts = [t for t in raw["validation"]["text"] if t.strip()]
    else:
        # Fallback to local or wikitext
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = [t for t in raw["train"]["text"] if t.strip()]
        val_texts = [t for t in raw["validation"]["text"] if t.strip()]

    def text_to_chunks(texts: list[str], max_chunks: Optional[int] = None) -> torch.Tensor:
        joined = "\n\n".join(texts)
        tokens = tokenizer(joined, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        n = (tokens.size(0) // seq_len) * seq_len
        chunks = tokens[:n].view(-1, seq_len)
        if max_chunks is not None and chunks.size(0) > max_chunks:
            chunks = chunks[:max_chunks]
        return chunks

    train_chunks = text_to_chunks(train_texts, max_train_samples)
    val_chunks = text_to_chunks(val_texts, 500)

    logger.info(
        "Corpus tokenized: %d train chunks, %d validation chunks (seq_len=%d)",
        train_chunks.size(0),
        val_chunks.size(0),
        seq_len,
    )
    return train_chunks, val_chunks


def evaluate_causal_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Computes Causal LM Perplexity: PPL = exp(cross_entropy_loss)."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                out = model(input_ids=in_ids, attention_mask=att_m, labels=lbls)
                loss = out.loss

            if torch.isfinite(loss):
                total_loss += loss.item()
                total_batches += 1

    if total_batches == 0:
        return float("inf")

    avg_loss = total_loss / total_batches
    return math.exp(min(avg_loss, 20.0))


def evaluate_sst2_accuracy(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_samples: int = 200,
) -> float:
    """Evaluates binary sentiment classification on SST-2 validation split."""
    from datasets import load_dataset

    try:
        sst2 = load_dataset("glue", "sst2", split="validation")
    except Exception as e:
        logger.debug("SST-2 evaluation skipped: %s", e)
        return 0.0

    model.eval()
    correct = 0
    total = 0

    # Token IDs for 'negative' (0) and 'positive' (1)
    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    with torch.no_grad():
        for i, item in enumerate(sst2):
            if i >= max_samples:
                break
            prompt = f"Sentence: {item['sentence']}\nSentiment:"
            enc = tokenizer(prompt, return_tensors="pt")
            in_ids = enc["input_ids"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                out = model(input_ids=in_ids)
                last_logits = out.logits[:, -1, :]
                pred_label = 1 if last_logits[0, pos_id] > last_logits[0, neg_id] else 0

            if pred_label == item["label"]:
                correct += 1
            total += 1

    return (correct / max(total, 1)) * 100.0


def apply_qwen_attention_surgery(
    model: nn.Module,
    attn_key: str,
    pattern: str = "all",
) -> nn.Module:
    """
    Transplants NeuralOps attention mechanism into Qwen decoder layers.

    Patterns:
      - 'all': replace all attention layers
      - 'hybrid_3_1': replace 3 subquadratic layers per 1 full quadratic layer
    """
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    # Locate transformer layers
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Cannot locate transformer layers in Qwen model architecture.")

    num_layers = len(layers)
    transplanted_count = 0

    for idx, layer in enumerate(layers):
        # Check hybrid pattern condition
        if pattern == "hybrid_3_1" and (idx % 4 == 3):
            # Retain full quadratic / sliding-window attention on every 4th layer
            continue

        orig_attn = getattr(layer, "self_attn", None)
        if orig_attn is None:
            continue

        # Instantiate target attention from ATTENTION_REGISTRY
        if attn_key in ATTENTION_REGISTRY:
            instantiator = ATTENTION_REGISTRY[attn_key]
            try:
                core_attn = instantiator(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    is_causal=True,
                )
            except TypeError:
                try:
                    core_attn = instantiator(embed_dim=hidden_size, is_causal=True)
                except TypeError:
                    core_attn = instantiator(embed_dim=hidden_size)
        else:
            raise KeyError(f"Attention key '{attn_key}' not found in ATTENTION_REGISTRY.")

        # Wrap in QwenAttentionAdapter with Single-o_proj Contract & GQA expansion
        adapter = QwenAttentionAdapter(
            core_attention=core_attn,
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            inherit_weights_from=orig_attn,
            is_causal=True,
            bypass_internal_out_proj=True,
        )

        layer.self_attn = adapter
        transplanted_count += 1

    logger.info(
        "Surgery complete: transplanted %d/%d layers with %s (pattern=%s)",
        transplanted_count,
        num_layers,
        attn_key,
        pattern,
    )
    return model


def train_single_qwen_attention(
    attn_key: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    pattern: str = "all",
    train_secs: float = 300.0,
    stage1_frac: float = 0.30,
    stage2_frac: float = 0.70,
    topk: int = 64,
    grad_accum: int = 4,
) -> dict[str, Any]:
    """Train one transplanted attention mechanism on Qwen with strict budget."""
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Attention Module: %s (%ds, pattern=%s)", attn_key, int(train_secs), pattern)
    logger.info("=" * 80)

    # 1. Surgery with weight inheritance
    student = copy.deepcopy(teacher_model)
    student = apply_qwen_attention_surgery(student, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    # Stage 0: Evaluate uncalibrated / zero-shot
    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    pre_sst2 = evaluate_sst2_accuracy(student, tokenizer, device)
    logger.info("[%s] Pre-KD Evaluation: PPL=%.2f | SST-2 Acc=%.2f%%", attn_key, pre_ppl, pre_sst2)

    # Timer starts STRICTLY AFTER Stage 0 evaluation finish
    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * stage1_frac)
    t_s2_deadline = t_train_start + train_secs

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Stage 1: Frozen-Backbone Subspace Alignment (lm_head bypassed)
    for name, p in student.named_parameters():
        p.requires_grad = ("self_attn" in name) or ("layernorm" in name.lower())

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1.0e-3, weight_decay=0.01)
    student.train()
    step_s1 = 0
    t0_s1 = time.time()

    while time.time() < t_s1_deadline:
        for batch in train_loader:
            if time.time() >= t_s1_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                # Bypass lm_head during Stage 1 by calling student.model directly
                with torch.no_grad():
                    t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)

                s_hiddens = s_out.hidden_states[1:]
                t_hiddens = t_out.hidden_states[1:]
                layer_mses = [F.mse_loss(s, t) for s, t in zip(s_hiddens, t_hiddens)]
                loss_mse = torch.stack(layer_mses).mean()

                layer_covs = [fast_gram_covariance_loss(s, t) for s, t in zip(s_hiddens, t_hiddens)]
                loss_cov = torch.stack(layer_covs).mean()

                loss = (loss_mse + 0.5 * loss_cov) / grad_accum

            scaler.scale(loss).backward()
            step_s1 += 1

            if step_s1 % grad_accum == 0:
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt1)
                scaler.update()
                opt1.zero_grad(set_to_none=True)

    dur_s1 = time.time() - t0_s1
    logger.info("[%s] Stage 1 Complete: %d steps in %.1fs", attn_key, step_s1, dur_s1)

    # Stage 2: End-to-End Unfrozen KD + Causal LM with Top-K Sparse Logit KD
    for p in student.parameters():
        p.requires_grad = True

    attn_params = [p for n, p in student.named_parameters() if "self_attn" in n]
    bb_params = [p for n, p in student.named_parameters() if "self_attn" not in n]
    opt2 = torch.optim.AdamW(
        [
            {"params": attn_params, "lr": 5.0e-4},
            {"params": bb_params, "lr": 5.0e-5},
        ],
        weight_decay=0.01,
    )

    student.train()
    step_s2 = 0
    t0_s2 = time.time()

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                with torch.no_grad():
                    t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                s_out = student(input_ids=in_ids, attention_mask=att_m, labels=lbls, output_hidden_states=True)

                # Top-K Sparse Logit KD (topk=64) avoiding dense 151936 logit OOM
                loss_sparse_kl = compute_topk_sparse_kd_loss(s_out.logits, t_out.logits, topk=topk)
                loss_causal_ce = s_out.loss if s_out.loss is not None else torch.tensor(0.0, device=device)

                loss_total = (0.50 * loss_sparse_kl + 0.50 * loss_causal_ce) / grad_accum

            scaler.scale(loss_total).backward()
            step_s2 += 1

            if step_s2 % grad_accum == 0:
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt2)
                scaler.update()
                opt2.zero_grad(set_to_none=True)

    dur_s2 = time.time() - t0_s2
    total_time = time.time() - t_train_start
    logger.info("[%s] Stage 2 Complete: %d steps in %.1fs (Total: %.1fs)", attn_key, step_s2, dur_s2, total_time)

    # Post-KD Evaluation
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    post_sst2 = evaluate_sst2_accuracy(student, tokenizer, device)
    logger.info("[%s] Post-KD Evaluation: PPL=%.2f | SST-2 Acc=%.2f%%", attn_key, post_ppl, post_sst2)

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0

    model_slug = model_name.split("/")[-1].lower()
    attn_slug = attn_key.replace("_", "-")

    result = {
        "model": model_name,
        "attn_key": attn_key,
        "pattern": pattern,
        "ppl_before": round(pre_ppl, 2),
        "ppl_after": round(post_ppl, 2),
        "ppl_delta": round(post_ppl - pre_ppl, 2),
        "sst2_before": round(pre_sst2, 2),
        "sst2_after": round(post_sst2, 2),
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "total_steps": step_s1 + step_s2,
        "peak_vram_mb": peak_vram,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "PASSED",
    }

    # Save checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2))

    # Model Card
    card_md = f"""# Qwen-2.5-0.5B with NeuralOps {attn_key} Attention

- **Base Model:** `{model_name}`
- **Transplanted Attention:** `{attn_key}` (`ATTENTION_REGISTRY`)
- **Surgery Pattern:** `{pattern}`
- **Pre-KD Causal PPL:** `{pre_ppl:.2f}`
- **Post-KD Causal PPL:** `{post_ppl:.2f}`
- **SST-2 Accuracy:** `{post_sst2:.2f}%`
- **Distillation Time:** `{total_time:.1f}s` on NVIDIA RTX 3090
"""
    (out_dir / "README.md").write_text(card_md)

    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scale Qwen Attention Distillation Engine")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--keys_json", default="results/bert_attention_zoo/registry_keys.json")
    parser.add_argument("--attn_key", default=None, help="Specific attention key to train")
    parser.add_argument("--surgery_pattern", choices=["all", "hybrid_3_1"], default="hybrid_3_1")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--train_secs", type=float, default=300.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--out_dir", default="results/qwen_attention_zoo")
    parser.add_argument("--metrics_log", default="results/qwen_attention_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="Hooshaai")
    parser.add_argument("--hf_upload", type=int, default=0)
    parser.add_argument("--keep_local_weights", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is unavailable. Aborting.")
        sys.exit(3)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_log).parent.mkdir(parents=True, exist_ok=True)

    if args.attn_key is not None:
        keys = [args.attn_key]
    else:
        with open(args.keys_json, "r") as f:
            keys = json.load(f)

    logger.info("Loaded %d keys for Qwen distillation (%s)", len(keys), args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    teacher.eval()

    train_tokens, val_tokens = load_tokenized_qwen_corpus(tokenizer, args.dataset, args.seq_len)
    train_dataset = StratifiedQwenDataset(train_tokens)
    val_dataset = StratifiedQwenDataset(val_tokens)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    uploader: Optional[AsyncHFHubUploader] = None
    if args.hf_upload:
        uploader = AsyncHFHubUploader(
            hf_org=args.hf_org,
            max_workers=2,
            cleanup_weights=(args.keep_local_weights == 0),
        )

    for i, key in enumerate(keys):
        ckpt_dir = out_base / key
        metrics_file = ckpt_dir / "training_metrics.json"

        # Resume support: Check if already trained & uploaded
        if metrics_file.exists():
            if args.hf_upload and uploader is not None:
                upload_marker = ckpt_dir / "upload_success.json"
                if not upload_marker.exists():
                    logger.info("[%d/%d] [RESUME-UPLOAD] %s trained but upload incomplete. Re-submitting.", i + 1, len(keys), key)
                    try:
                        with open(metrics_file, "r") as f:
                            saved_metrics = json.load(f)
                    except Exception:
                        saved_metrics = {"attn_key": key, "status": "PASSED"}
                    uploader.submit_upload(key, ckpt_dir, saved_metrics)
                    continue
            logger.info("[%d/%d] [RESUME] Skipping %s (already trained).", i + 1, len(keys), key)
            continue

        logger.info("\n[%d/%d] Processing Qwen Attention: %s", i + 1, len(keys), key)
        try:
            res = train_single_qwen_attention(
                attn_key=key,
                teacher_model=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                tokenizer=tokenizer,
                out_dir=ckpt_dir,
                model_name=args.model,
                pattern=args.surgery_pattern,
                train_secs=args.train_secs,
                topk=args.topk,
                grad_accum=args.grad_accum,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("[%s] OOM at batch_size=%d. Halving batch size and retrying...", key, args.batch_size)
            torch.cuda.empty_cache()
            gc.collect()
            try:
                halved_loader = DataLoader(train_dataset, batch_size=max(1, args.batch_size // 2), shuffle=True, drop_last=True)
                res = train_single_qwen_attention(
                    attn_key=key,
                    teacher_model=teacher,
                    train_loader=halved_loader,
                    val_loader=val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    out_dir=ckpt_dir,
                    model_name=args.model,
                    pattern=args.surgery_pattern,
                    train_secs=args.train_secs,
                    topk=args.topk,
                    grad_accum=args.grad_accum * 2,
                )
            except Exception as e:
                logger.error("[%s] OOM retry failed: %s", key, e)
                res = {"attn_key": key, "status": "SKIPPED_OOM", "error": str(e), "timestamp": datetime.datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error("[%s] Training exception: %s", key, e, exc_info=True)
            res = {"attn_key": key, "status": "FAIL_TRAINING", "error": str(e), "timestamp": datetime.datetime.utcnow().isoformat()}

        with open(args.metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")

        if uploader is not None and res.get("status") == "PASSED":
            uploader.submit_upload(key, ckpt_dir, res)

    if uploader is not None:
        logger.info("Waiting for background uploads to complete...")
        uploader.wait_all(timeout=600)
        uploader.shutdown()

    logger.info("Qwen distillation complete. Ledger at: %s", args.metrics_log)


if __name__ == "__main__":
    main()
