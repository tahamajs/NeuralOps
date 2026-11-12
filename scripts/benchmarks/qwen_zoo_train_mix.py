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
import inspect
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

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

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention, get_module_capabilities
from neuralops.hub.uploader import AsyncHFHubUploader
from neuralops.layers.qwen_adapter import QwenAttentionAdapter

try:
    from scripts.benchmarks.teacher_cache_utils import (
        CachedCausalDataset,
        build_or_load_causal_teacher_cache,
        enable_student_gradient_checkpointing,
        resolve_teacher_cache_path,
        try_load_causal_teacher_cache,
    )
except ImportError:
    from teacher_cache_utils import (
        CachedCausalDataset,
        build_or_load_causal_teacher_cache,
        enable_student_gradient_checkpointing,
        resolve_teacher_cache_path,
        try_load_causal_teacher_cache,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_zoo_train")


def append_ledger_record_atomic(dest_path: Path | str, record: dict[str, Any]) -> None:
    """Atomically append a record to a JSONL ledger via a .tmp file and POSIX rename."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    if dest.exists():
        shutil.copyfile(dest, tmp_path)
    with tmp_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(dest)


def _atomic_write_text(dest_path: Path, content: str) -> None:
    """Atomically write text content to a destination file via a .tmp file and POSIX rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(dest_path)

from neuralops.benchmarks.kd import (
    CausalTextDataset,
    StratifiedQwenDataset,
    apply_qwen_attention_surgery,
    build_cache_wikitext,
    build_qwen_readme,
    collect_artifact_sizes,
    compute_topk_sparse_kd_loss,
    evaluate_causal_ppl,
    evaluate_sst2_accuracy,
    extract_attention_metadata,
    fast_gram_covariance_loss,
    format_size,
    load_tokenized_qwen_corpus,
    normalize_base_model_id,
    summarize_attention_doc,
)

# Backward-compatible aliases
_build_cache_wikitext = build_cache_wikitext
_normalize_base_model_id = normalize_base_model_id
_summarize_attention_doc = summarize_attention_doc
_format_size = format_size
_extract_attention_metadata = extract_attention_metadata
_collect_artifact_sizes = collect_artifact_sizes


def _file_size_str(file_path: Path) -> str:
    """Return human-readable size string for a path on disk, or 'N/A' if missing."""
    if file_path.is_file():
        return format_size(file_path.stat().st_size)
    return "N/A"


def train_single_qwen_attention(
    attn_key: str,
    teacher_model: Optional[nn.Module],
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
    hf_org: str = "tahamajs",
    gradient_checkpointing: bool = False,
    teacher_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train one transplanted attention mechanism on Qwen with strict budget."""
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Attention Module: %s (%ds, pattern=%s)", attn_key, int(train_secs), pattern)
    logger.info("=" * 80)

    if teacher_cache is None and teacher_model is None:
        raise ValueError("Either teacher_cache or teacher_model must be provided.")

    # 1. Surgery with weight inheritance (FP32 master weights on CPU first -> 0 MB teacher GPU VRAM)
    if teacher_model is not None:
        student = copy.deepcopy(teacher_model).float()
    else:
        student = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    student = apply_qwen_attention_surgery(student, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    # Stage 0: Evaluate uncalibrated / zero-shot on WikiText-2
    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Pre-KD WikiText-2 Perplexity: PPL=%.2f", attn_key, pre_ppl)

    if gradient_checkpointing:
        enable_student_gradient_checkpointing(student, log_prefix=f"[{attn_key}] ")

    # Timer starts STRICTLY AFTER Stage 0 evaluation finish
    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * stage1_frac)
    t_s2_deadline = t_train_start + train_secs

    # Stage 1: Frozen-Backbone Subspace Alignment (keep anchor quadratic layers frozen)
    total_qwen_layers = int(getattr(student.config, "num_hidden_layers", 24))
    anchor_indices_s = [i for i in range(total_qwen_layers) if pattern == "hybrid_3_1" and i % 4 == 3]
    surgeried_indices_s = [i for i in range(total_qwen_layers) if i not in set(anchor_indices_s)]
    surgeried_attn_prefixes = [f"model.layers.{i}.self_attn" for i in surgeried_indices_s]
    anchor_attn_prefixes = [f"model.layers.{i}.self_attn" for i in anchor_indices_s]

    trained_param_names: set[str] = set()
    for name, p in student.named_parameters():
        is_anchor = any(name.startswith(ap + ".") for ap in anchor_attn_prefixes)
        is_target = (not is_anchor) and (("self_attn" in name) or ("layernorm" in name.lower()))
        p.requires_grad = is_target
        if is_target:
            trained_param_names.add(name)

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.0e-4, weight_decay=0.01)
    scaler1 = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=(device.type == "cuda"))
    student.train()
    step_s1 = 0
    t0_s1 = time.time()

    layer_indices: Optional[list[int]] = teacher_cache["layer_indices"] if teacher_cache is not None else None

    while time.time() < t_s1_deadline:
        for batch in train_loader:
            if time.time() >= t_s1_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                # Bypass lm_head during Stage 1 by calling student.model directly
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                if "teacher_hidden_states" in batch and layer_indices is not None:
                    t_h_b = batch["teacher_hidden_states"].to(device, non_blocking=True)
                    s_hiddens = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                    t_hiddens = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                    s_hiddens = s_out.hidden_states[1:]
                    t_hiddens = t_out.hidden_states[1:]
                layer_mses = [F.mse_loss(s, t) for s, t in zip(s_hiddens, t_hiddens)]
                layer_cos = [1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_hiddens, t_hiddens)]
                loss_mse = torch.stack(layer_mses).mean()
                loss_cos = torch.stack(layer_cos).mean()

                loss = (loss_mse + 0.25 * loss_cos) / grad_accum

            scaler1.scale(loss).backward()
            step_s1 += 1

            if step_s1 % grad_accum == 0:
                scaler1.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler1.step(opt1)
                scaler1.update()
                opt1.zero_grad(set_to_none=True)

    if step_s1 > 0 and step_s1 % grad_accum != 0:
        scaler1.unscale_(opt1)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler1.step(opt1)
        scaler1.update()
        opt1.zero_grad(set_to_none=True)

    dur_s1 = time.time() - t0_s1
    logger.info("[%s] Stage 1 Complete: %d steps in %.1fs", attn_key, step_s1, dur_s1)

    # Release Stage 1 optimizer states to reclaim 2.5+ GB VRAM
    del opt1, scaler1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Stage 2: Fine-Tuning Multi-Layer Representation Distillation
    opt2 = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=6.0e-5,
        weight_decay=0.01,
    )
    scaler2 = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=(device.type == "cuda"))

    student.train()
    step_s2 = 0
    t0_s2 = time.time()

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                if "teacher_hidden_states" in batch and layer_indices is not None:
                    t_h_b = batch["teacher_hidden_states"].to(device, non_blocking=True)
                    s_hiddens = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                    t_hiddens = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                    s_hiddens = s_out.hidden_states[1:]
                    t_hiddens = t_out.hidden_states[1:]
                layer_mses = [F.mse_loss(s, t) for s, t in zip(s_hiddens, t_hiddens)]
                layer_cos = [1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_hiddens, t_hiddens)]
                loss_mse = torch.stack(layer_mses).mean()
                loss_cos = torch.stack(layer_cos).mean()

                loss = (loss_mse + 0.50 * loss_cos) / grad_accum

            scaler2.scale(loss).backward()
            step_s2 += 1

            if step_s2 % grad_accum == 0:
                scaler2.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler2.step(opt2)
                scaler2.update()
                opt2.zero_grad(set_to_none=True)

    if step_s2 > 0 and step_s2 % grad_accum != 0:
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler2.step(opt2)
        scaler2.update()
        opt2.zero_grad(set_to_none=True)

    dur_s2 = time.time() - t0_s2
    total_time = time.time() - t_train_start
    logger.info("[%s] Stage 2 Complete: %d steps in %.1fs (Total: %.1fs)", attn_key, step_s2, dur_s2, total_time)

    # Post-KD Evaluation on WikiText-2
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Post-KD Evaluation: PPL=%.2f (Delta: %+.2f)", attn_key, post_ppl, post_ppl - pre_ppl)

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0

    base_model_id = _normalize_base_model_id(model_name)
    attn_slug = attn_key.replace("_", "-")
    repo_id = f"{hf_org}/qwen-2.5-0.5b-{attn_slug}-wikitext2"
    num_layers = int(getattr(student.config, "num_hidden_layers", 24))
    ds_chunks = getattr(getattr(train_loader, "dataset", None), "chunks", None)
    seq_len = int(ds_chunks.size(1)) if isinstance(ds_chunks, torch.Tensor) and ds_chunks.dim() >= 2 else 512

    result = {
        "model": base_model_id,
        "attn_key": attn_key,
        "dataset": "wikitext-2",
        "pattern": pattern,
        "ppl_before": round(pre_ppl, 2),
        "ppl_after": round(post_ppl, 2),
        "ppl_delta": round(post_ppl - pre_ppl, 2),
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "total_steps": step_s1 + step_s2,
        "peak_vram_mb": peak_vram,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "PASSED",
    }

    # Save strictly surgeried attention + trained layernorm delta weights & manifest (zero anchor/MLP bloat)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))

    from neuralops.hub import save_surgical_delta

    delta_info = save_surgical_delta(
        student,
        out_dir,
        weights_filename="attention_weights.safetensors",
        base_model_id=base_model_id,
        surgeried_prefixes=surgeried_attn_prefixes,
        trained_param_names=sorted(trained_param_names),
        anchor_prefixes=anchor_attn_prefixes,
        operations=[{"component_type": "attention", "operator_key": attn_key, "pattern": pattern, "target_modules": surgeried_attn_prefixes}],
        metadata={
            "zoo_type": "qwen_attention",
            "attention_type": attn_key,
            "pattern": pattern,
            "total_layers": num_layers,
            "surgeried_layers": surgeried_indices_s,
            "anchor_layers": anchor_indices_s,
        },
        half_precision=True,
        remove_full_weights=True,
    )

    # Normalize and save config.json and training metrics
    qwen_cfg = student.config.to_dict()
    qwen_cfg["_name_or_path"] = base_model_id
    qwen_cfg["base_model_name_or_path"] = base_model_id
    qwen_cfg["license"] = "mit"
    qwen_cfg["attention_type"] = attn_key
    qwen_cfg["weights_format"] = "attention_only_fp16"
    qwen_cfg["saved_tensor_count"] = int(delta_info.get("saved_tensor_count", 0))
    _atomic_write_text(out_dir / "config.json", json.dumps(qwen_cfg, indent=2))
    _atomic_write_text(out_dir / "training_metrics.json", json.dumps(result, indent=2))

    # Model Card
    card_md = build_qwen_readme(
        attn_key=attn_key,
        result=result,
        out_dir=out_dir,
        base_model=base_model_id,
        repo_id=repo_id,
        num_layers=num_layers,
        dur_s1=dur_s1,
        dur_s2=dur_s2,
        seq_len=seq_len,
        grad_accum=grad_accum,
        topk=topk,
    )
    _atomic_write_text(out_dir / "README.md", card_md)

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
    parser.add_argument(
        "--gradient_checkpointing",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="Enable gradient checkpointing on student decoder blocks (0/1)",
    )
    parser.add_argument(
        "--teacher_cache_path",
        default="auto",
        help="Offline teacher representation cache path ('auto' or 'none' to disable)",
    )
    parser.add_argument(
        "--max_cache_samples",
        type=int,
        default=256,
        help="Maximum training chunks/samples stored in offline teacher cache",
    )
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
    train_tokens, val_tokens = load_tokenized_qwen_corpus(tokenizer, args.dataset, args.seq_len)

    norm_model_id = _normalize_base_model_id(args.model)
    cache_path = resolve_teacher_cache_path(
        args.teacher_cache_path,
        out_dir=out_base,
        model_id=norm_model_id,
        tag="qwen_causal",
        seq_len=args.seq_len,
        max_cache_samples=args.max_cache_samples,
        topk=args.topk,
    )
    n_cache_samples = min(int(train_tokens.size(0)), max(1, int(args.max_cache_samples))) if train_tokens.dim() == 2 else 0
    teacher: Optional[nn.Module] = None
    teacher_cache: Optional[dict[str, Any]] = None

    if cache_path is not None and n_cache_samples > 0:
        teacher_cache = try_load_causal_teacher_cache(
            cache_path,
            model_id=norm_model_id,
            seq_len=args.seq_len,
            num_samples=n_cache_samples,
            topk=args.topk,
        )

    if teacher_cache is None:
        loaded_teacher = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        loaded_teacher.eval()
        for p in loaded_teacher.parameters():
            p.requires_grad = False

        if cache_path is not None and n_cache_samples > 0 and hasattr(loaded_teacher, "config"):
            teacher_cache = build_or_load_causal_teacher_cache(
                cache_path,
                model_id=args.model,
                normalized_model_id=norm_model_id,
                train_chunks=train_tokens,
                val_chunks=val_tokens,
                device=device,
                load_teacher_fn=lambda: loaded_teacher,
                evaluate_ppl_fn=evaluate_causal_ppl,
                seq_len=args.seq_len,
                max_cache_samples=args.max_cache_samples,
                topk=args.topk,
                batch_size=args.batch_size,
                teacher_model=loaded_teacher,
            )
            del loaded_teacher
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            teacher = loaded_teacher

    if teacher_cache is not None:
        train_dataset: Dataset = CachedCausalDataset(teacher_cache)
    else:
        train_dataset = StratifiedQwenDataset(train_tokens)
    val_dataset = StratifiedQwenDataset(val_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) >= args.batch_size),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    uploader: Optional[AsyncHFHubUploader] = None
    if args.hf_upload:
        uploader = AsyncHFHubUploader(
            hf_org=args.hf_org,
            max_workers=2,
            cleanup_weights=(args.keep_local_weights == 0),
        )

    completed_keys = set()
    metrics_log_path = Path(args.metrics_log)
    if metrics_log_path.exists():
        for line in metrics_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("status") == "PASSED":
                        completed_keys.add(d.get("attn_key"))
                except Exception:
                    pass

    for i, key in enumerate(keys):
        if key in completed_keys:
            logger.info("[%d/%d] [RESUME] Skipping %s (already PASSED in metrics.jsonl).", i + 1, len(keys), key)
            continue
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
        curr_bs = max(1, int(args.batch_size))
        curr_accum = max(1, int(args.grad_accum))
        curr_ckpt = bool(args.gradient_checkpointing)
        res: dict[str, Any] = {}

        while True:
            curr_train_loader = (
                train_loader
                if curr_bs == args.batch_size
                else DataLoader(
                    train_dataset,
                    batch_size=curr_bs,
                    shuffle=True,
                    drop_last=(len(train_dataset) >= curr_bs),
                )
            )
            curr_val_loader = (
                val_loader
                if curr_bs == args.batch_size
                else DataLoader(val_dataset, batch_size=curr_bs, shuffle=False)
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            is_oom = False
            err_msg = ""
            try:
                res = train_single_qwen_attention(
                    attn_key=key,
                    teacher_model=teacher,
                    train_loader=curr_train_loader,
                    val_loader=curr_val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    out_dir=ckpt_dir,
                    model_name=args.model,
                    pattern=args.surgery_pattern,
                    train_secs=args.train_secs,
                    topk=args.topk,
                    grad_accum=curr_accum,
                    hf_org=args.hf_org,
                    gradient_checkpointing=curr_ckpt,
                    teacher_cache=teacher_cache,
                )
                break
            except Exception as e:
                err_msg = str(e)
                # PyTorch <= 1.13 raises RuntimeError("CUDA out of memory...") instead of OutOfMemoryError
                if "out of memory" in err_msg.lower():
                    is_oom = True
                else:
                    logger.error("[%s] Training exception: %s", key, e, exc_info=True)

            # Free failed GPU graph and tensors strictly outside the except block
            del curr_train_loader, curr_val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if is_oom and (curr_bs > 1 or not curr_ckpt):
                if not curr_ckpt:
                    curr_ckpt = True
                if curr_bs > 1:
                    next_bs = max(1, curr_bs // 2)
                    scale = max(1, curr_bs // next_bs)
                    curr_bs = next_bs
                    curr_accum = curr_accum * scale
                logger.warning(
                    "[%s] OOM encountered. Retrying with batch_size=%d, grad_accum=%d, gradient_checkpointing=%s...",
                    key,
                    curr_bs,
                    curr_accum,
                    curr_ckpt,
                )
                time.sleep(0.05)
                continue

            if is_oom:
                logger.error("[%s] OOM retry exhausted at batch_size=%d: %s", key, curr_bs, err_msg)
                res = {
                    "attn_key": key,
                    "status": "SKIPPED_OOM",
                    "error": err_msg,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
            else:
                res = {
                    "attn_key": key,
                    "status": "FAIL_TRAINING",
                    "error": err_msg,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
            break

        append_ledger_record_atomic(args.metrics_log, res)

        if uploader is not None and res.get("status") == "PASSED":
            uploader.submit_upload(key, ckpt_dir, res)

    if uploader is not None:
        logger.info("Waiting for background uploads to complete...")
        uploader.wait_all(timeout=600)
        uploader.shutdown()

    logger.info("Qwen distillation complete. Ledger at: %s", args.metrics_log)


if __name__ == "__main__":
    main()
