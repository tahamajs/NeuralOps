#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Master Track Qwen2.5-0.5B 82-Compressor Zoo Distillation Engine.

Compresses Qwen/Qwen2.5-0.5B using NeuralOps compressors from
results/compressor_zoo/registry_keys.json, measures exact parameter reduction
and sparsity, heals distortion via 2-stage Knowledge Distillation (Top-K Sparse
Logit KD + Causal CE + Layerwise Hidden-State MSE) from offline RAM caches, and
exports compact compressor_only_fp16 safetensors + Hugging Face Model Cards.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch 1.12 compatibility with modern HuggingFace Accelerate
try:
    _orig_named_parameters = nn.Module.named_parameters

    def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)

    nn.Module.named_parameters = _compat_named_parameters
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neuralops.core.compression_pipeline import compress_model
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor
from neuralops.hub.uploader import AsyncHFHubUploader
from scripts.benchmarks.qwen_zoo_train_mix import (
    _build_cache_wikitext,
    _collect_artifact_sizes,
    _normalize_base_model_id,
    compute_topk_sparse_kd_loss,
    evaluate_causal_ppl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_compressor_zoo_train")


def _compute_sparsity_pct(model: nn.Module) -> float:
    """Measure exact percentage of zero elements across floating weights and buffers."""
    total_elems = 0
    zero_elems = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point() and tensor.numel() > 0:
            total_elems += tensor.numel()
            zero_elems += int((tensor == 0).sum().item())
    if total_elems == 0:
        return 0.0
    return round((zero_elems / total_elems) * 100.0, 2)


def _save_compressed_weights(student: nn.Module, out_dir: Path) -> int:
    """Persist compressed / decomposed layer parameters in compact FP16 safetensors."""
    from safetensors.torch import save_file

    comp_state: dict[str, torch.Tensor] = {}
    keywords = (
        "compress",
        "lora",
        "adapter",
        "factor",
        "svd",
        "quant",
        "mask",
        "scale",
        "zero",
        "u_proj",
        "v_proj",
        "weight_u",
        "weight_v",
        "core",
    )
    for k, v in student.state_dict().items():
        k_lower = k.lower()
        if any(term in k_lower for term in keywords):
            comp_state[k] = (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()).contiguous()

    # Include compressed/modified attention & MLP weights so checkpoints are self-contained deltas.
    for k, v in student.state_dict().items():
        if k not in comp_state and (".self_attn." in k or ".mlp." in k):
            comp_state[k] = (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()).contiguous()

    save_file(comp_state, str(out_dir / "attention_weights.safetensors"))

    for heavy_name in ["pytorch_model.bin", "model.safetensors", "optimizer.pt"]:
        h_path = out_dir / heavy_name
        if h_path.exists():
            os.remove(h_path)

    return len(comp_state)


def _build_qwen_comp_readme(
    *,
    repo_id: str,
    base_model: str,
    comp_key: str,
    rank: int,
    orig_params: int,
    comp_params: int,
    param_reduction: float,
    sparsity_pct: float,
    pre_ppl: float | None,
    post_ppl: float | None,
    steps_s1: int,
    steps_s2: int,
    train_secs: float,
    peak_vram_mb: int,
    out_dir: Path,
) -> str:
    """Construct a complete Hugging Face Model Card for compressed Qwen2.5-0.5B."""
    base_id = _normalize_base_model_id(base_model)
    cls_obj = COMPRESSOR_REGISTRY.get(comp_key)
    cls_doc = ((getattr(cls_obj, "__doc__", None) or "").strip().splitlines()[0]
               if cls_obj is not None else f"NeuralOps `{comp_key}` compressor.")
    pre_s = f"{pre_ppl:.2f}" if pre_ppl is not None and math.isfinite(pre_ppl) else "N/A"
    post_s = f"{post_ppl:.2f}" if post_ppl is not None and math.isfinite(post_ppl) else "N/A"
    delta_s = (
        f"{post_ppl - pre_ppl:+.2f}"
        if (pre_ppl is not None and post_ppl is not None and math.isfinite(pre_ppl) and math.isfinite(post_ppl))
        else "N/A"
    )
    artifacts, total_size = _collect_artifact_sizes(out_dir)
    artifact_rows = "\n".join(f"| `{name}` | {sz} |" for name, sz in artifacts) or "| `attention_weights.safetensors` | FP16 |"

    yaml_results = ""
    if post_ppl is not None and math.isfinite(post_ppl):
        yaml_results = f"""model-index:
  - name: {repo_id}
    results:
      - task:
          type: text-generation
          name: Causal Language Modeling
        dataset:
          type: Salesforce/wikitext
          name: WikiText-2 (wikitext-2-raw-v1)
          split: validation
        metrics:
          - type: perplexity
            name: Validation Perplexity
            value: {post_ppl:.2f}
          - type: custom
            name: Parameter Reduction Ratio
            value: {param_reduction:.3f}"""

    return f"""---
library_name: neuralops
base_model: {base_id}
license: mit
pipeline_tag: text-generation
language:
  - en
datasets:
  - Salesforce/wikitext
tags:
  - neuralops
  - model-compression
  - knowledge-distillation
  - qwen2.5
  - {comp_key.replace('_', '-')}
{yaml_results}
---

# `{repo_id}`

> Compressed **`{base_id}`** checkpoint using NeuralOps **`{comp_key}`** (`rank={rank}`) and healed with 2-Stage Sparse Logit & Hidden-State Knowledge Distillation on **WikiText-2 (`Salesforce/wikitext`)**.

## Compression & Healing Summary

{cls_doc}

| Metric | Value |
|---|---|
| **Base Model** | [`{base_id}`](https://huggingface.co/{base_id}) |
| **Compressor Key** | `{comp_key}` |
| **Target Rank** | `{rank}` |
| **Original Parameters** | `{orig_params:,}` |
| **Compressed Parameters** | `{comp_params:,}` |
| **Parameter Reduction Ratio** | **{param_reduction:.3f}x** |
| **Weight Sparsity** | **{sparsity_pct:.2f}%** |
| **Pre-KD WikiText-2 PPL** | `{pre_s}` |
| **Post-KD WikiText-2 PPL** | **{post_s}** (`{delta_s}`) |
| **Distillation Steps (S1 + S2)** | `{steps_s1} + {steps_s2}` (`{train_secs:.1f}s`) |
| **Peak GPU VRAM** | `{peak_vram_mb} MB` |
| **Weights Format** | `compressor_only_fp16` (`attention_weights.safetensors`) |

## Checkpoint Artifacts ({total_size})

| File | Size |
|---|---|
{artifact_rows}

## Quick Start

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.core.compression_pipeline import compress_model

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained("{base_id}", torch_dtype="auto")
model = compress_model(model, {{"compressors": [{{"method": "{comp_key}", "rank": {rank}, "force_rank": True, "use_strs": False}}]}})

deltas = load_file(hf_hub_download(repo_id, "attention_weights.safetensors"))
model.load_state_dict(deltas, strict=False)
```

## License

Licensed under the **MIT License** (same as [NeuralOps](https://github.com/tahamajs/NeuralOps)).
"""


def train_single_qwen_compressor(
    comp_key: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    repo_id: str = "tahamajs/qwen-2.5-0.5b-comp-unknown-wikitext2",
    train_secs: float = 60.0,
    topk: int = 64,
    rank: int = 128,
    grad_accum: int = 8,
) -> dict[str, Any]:
    """Apply `comp_key` compression on CPU FP32, then heal on GPU using AMP + Top-K Sparse KD."""
    t_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Compressor: %s (%.1fs, rank=%d)", comp_key, train_secs, rank)
    logger.info("=" * 80)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Instantiate fresh FP32 base on CPU so SVD/QR/decomposition compressors run cleanly
    # and trainable master parameters stay FP32 for PyTorch 1.12 GradScaler compatibility.
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    orig_params = sum(p.numel() for p in student.parameters())

    try:
        cfg = {"compressors": [{"method": comp_key, "rank": rank, "force_rank": True, "use_strs": False}]}
        student = compress_model(student, cfg)
    except Exception as e:
        logger.warning("[%s] compress_model failed, falling back to factory: %s", comp_key, e)
        inst = create_compressor(comp_key, model=student, rank=rank)
        if hasattr(inst, "compress"):
            res = inst.compress()
            student = res[0] if isinstance(res, tuple) else res

    # Normalize any compressed float16 buffers back to FP32 if trainable so GradScaler works safely
    for p in student.parameters():
        if p.dtype == torch.float16:
            p.data = p.data.float()

    student = student.to(device)
    comp_params = sum(p.numel() for p in student.parameters())
    param_reduction = orig_params / max(comp_params, 1)
    sparsity_pct = _compute_sparsity_pct(student)
    logger.info(
        "[%s] Params: %d -> %d (Ratio=%.3fx | Sparsity=%.2f%%)",
        comp_key, orig_params, comp_params, param_reduction, sparsity_pct,
    )

    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Pre-KD WikiText-2 PPL: %.2f", comp_key, pre_ppl)

    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * 0.30)
    t_s2_deadline = t_train_start + train_secs

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Stage 1: Frozen Layer Alignment (compressed/attn/ln parameters)
    trainable_s1 = []
    for name, p in student.named_parameters():
        is_target = any(k in name.lower() for k in ("compress", "svd", "factor", "lora", "u_proj", "v_proj", "self_attn", "layernorm"))
        p.requires_grad = bool(is_target and p.is_floating_point())
        if p.requires_grad:
            trainable_s1.append(p)
    if not trainable_s1:
        for p in student.parameters():
            if p.is_floating_point():
                p.requires_grad = True
                trainable_s1.append(p)

    opt1 = torch.optim.AdamW(trainable_s1, lr=8.0e-4, weight_decay=0.01)
    student.train()
    step_s1 = 0

    while time.time() < t_s1_deadline:
        for batch in train_loader:
            if time.time() >= t_s1_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_out = student(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_h = s_out.hidden_states[1:]
                t_h = t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(sh.float(), th.float()) for sh, th in zip(s_h, t_h)]).mean()
                loss = loss_mse / grad_accum
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            step_s1 += 1
            if step_s1 % grad_accum == 0:
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(trainable_s1, 1.0)
                scaler.step(opt1)
                scaler.update()
                opt1.zero_grad(set_to_none=True)

    # Flush Stage 1 leftovers
    if step_s1 % grad_accum != 0:
        scaler.unscale_(opt1)
        scaler.step(opt1)
        scaler.update()
        opt1.zero_grad(set_to_none=True)
    del opt1

    # Stage 2: End-to-End Unfrozen Top-K Sparse Logit KD + Causal LM CE
    trainable_s2 = []
    for p in student.parameters():
        if p.is_floating_point():
            p.requires_grad = True
            trainable_s2.append(p)

    opt2 = torch.optim.AdamW(trainable_s2, lr=1.0e-4, weight_decay=0.01)
    student.train()
    step_s2 = 0

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, use_cache=False)
                s_out = student(input_ids=in_ids, attention_mask=att_m, labels=lbls, use_cache=False)
                loss_sparse = compute_topk_sparse_kd_loss(s_out.logits.float(), t_out.logits.float(), topk=topk)
                loss_ce = s_out.loss.float() if s_out.loss is not None else torch.tensor(0.0, device=device)
                loss = (0.50 * loss_sparse + 0.50 * loss_ce) / grad_accum
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            step_s2 += 1
            if step_s2 % grad_accum == 0:
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(trainable_s2, 1.0)
                scaler.step(opt2)
                scaler.update()
                opt2.zero_grad(set_to_none=True)

    if step_s2 % grad_accum != 0:
        scaler.unscale_(opt2)
        scaler.step(opt2)
        scaler.update()
        opt2.zero_grad(set_to_none=True)
    del opt2

    total_time = time.time() - t_start
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0
    final_sparsity = _compute_sparsity_pct(student)

    is_finite_ok = math.isfinite(post_ppl)
    pre_val = round(pre_ppl, 2) if math.isfinite(pre_ppl) else None
    post_val = round(post_ppl, 2) if is_finite_ok else None
    delta_val = round(post_ppl - pre_ppl, 2) if (is_finite_ok and math.isfinite(pre_ppl)) else None
    status = "PASSED" if is_finite_ok else "FAIL_NUMERICS"

    result: dict[str, Any] = {
        "model": _normalize_base_model_id(model_name),
        "compressor": comp_key,
        "rank": rank,
        "param_reduction": round(param_reduction, 3),
        "sparsity_pct": final_sparsity,
        "orig_params": orig_params,
        "comp_params": comp_params,
        "ppl_before": pre_val,
        "ppl_after": post_val,
        "ppl_delta": delta_val,
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "peak_vram_mb": peak_vram,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": status,
    }

    if status == "PASSED":
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        num_tensors = _save_compressed_weights(student, out_dir)

        cfg_out = student.config.to_dict()
        cfg_out["_name_or_path"] = _normalize_base_model_id(model_name)
        cfg_out["compressor_key"] = comp_key
        cfg_out["compression_rank"] = rank
        cfg_out["weights_format"] = "compressor_only_fp16"
        cfg_out["saved_tensor_count"] = num_tensors
        (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
        (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        readme_text = _build_qwen_comp_readme(
            repo_id=repo_id,
            base_model=model_name,
            comp_key=comp_key,
            rank=rank,
            orig_params=orig_params,
            comp_params=comp_params,
            param_reduction=param_reduction,
            sparsity_pct=final_sparsity,
            pre_ppl=pre_val,
            post_ppl=post_val,
            steps_s1=step_s1,
            steps_s2=step_s2,
            train_secs=total_time,
            peak_vram_mb=peak_vram,
            out_dir=out_dir,
        )
        (out_dir / "README.md").write_text(readme_text, encoding="utf-8")
        logger.info("[%s] Saved %d compressed tensors + README to %s (PPL: %s -> %s)", comp_key, num_tensors, out_dir, pre_val, post_val)

    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Qwen2.5-0.5B 82-Compressor Zoo Distillation")
    parser.add_argument("--keys_json", default="results/compressor_zoo/registry_keys.json")
    parser.add_argument("--comp_key", default="", help="Optional single compressor key for process isolation.")
    parser.add_argument("--model", default="/dev/shm/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--train_secs", type=float, default=60.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/qwen_compressor_zoo")
    parser.add_argument("--metrics_log", default="/home/tahamajlesi.ut/NeuralOps/results/qwen_compressor_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--hf_upload", type=int, default=0)
    parser.add_argument("--keep_local_weights", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available.")
        sys.exit(3)

    if args.comp_key:
        comp_keys = [args.comp_key]
    else:
        keys_path = Path(args.keys_json)
        if not keys_path.exists():
            logger.error("Keys JSON not found: %s", keys_path)
            sys.exit(2)
        comp_keys = json.loads(keys_path.read_text(encoding="utf-8"))

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(args.metrics_log)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    done_passed: set[str] = set()
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "PASSED" and rec.get("compressor"):
                        done_passed.add(rec["compressor"])
                except Exception:
                    pass

    pending_keys = [
        k for k in comp_keys
        if k not in done_passed and not (out_base / k / "training_metrics.json").exists()
    ]
    if not pending_keys:
        logger.info("All requested Qwen compressor keys already completed. Exiting cleanly.")
        os._exit(0)

    hf_token = os.environ.get("HF_TOKEN")
    uploader = None
    if args.hf_upload and hf_token:
        uploader = AsyncHFHubUploader(token=hf_token, org=args.hf_org)
        uploader.start()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_ds, val_ds = _build_cache_wikitext(args.dataset, tokenizer, args.seq_len, cache_dir="/tmp")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    for i, comp_key in enumerate(pending_keys):
        logger.info("\n[%d/%d] Processing Qwen Compressor: %s", i + 1, len(pending_keys), comp_key)
        ckpt_dir = out_base / comp_key
        repo_id = f"{args.hf_org}/qwen-2.5-0.5b-comp-{comp_key.replace('_', '-')}-wikitext2"

        oom_retry = False
        res: dict[str, Any] | None = None
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
                repo_id=repo_id,
                train_secs=args.train_secs,
                topk=args.topk,
                rank=args.rank,
                grad_accum=args.grad_accum,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and args.batch_size > 1:
                oom_retry = True
                logger.warning("[%s] CUDA OOM at bs=%d -> freeing & retrying bs=%d", comp_key, args.batch_size, max(1, args.batch_size // 2))
            else:
                logger.error("[%s] Runtime failure: %s", comp_key, e, exc_info=True)
                res = {
                    "model": _normalize_base_model_id(args.model),
                    "compressor": comp_key,
                    "status": "FAIL_TRAINING",
                    "error": str(e)[:240],
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
        except Exception as e:
            logger.error("[%s] Training failed: %s", comp_key, e, exc_info=True)
            res = {
                "model": _normalize_base_model_id(args.model),
                "compressor": comp_key,
                "status": "FAIL_TRAINING",
                "error": str(e)[:240],
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }

        if oom_retry:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            low_bs = max(1, args.batch_size // 2)
            low_accum = args.grad_accum * max(1, args.batch_size // low_bs)
            low_train_loader = DataLoader(train_ds, batch_size=low_bs, shuffle=True, pin_memory=(device.type == "cuda"))
            low_val_loader = DataLoader(val_ds, batch_size=low_bs, shuffle=False, pin_memory=(device.type == "cuda"))
            try:
                res = train_single_qwen_compressor(
                    comp_key=comp_key,
                    teacher_model=teacher,
                    train_loader=low_train_loader,
                    val_loader=low_val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    out_dir=ckpt_dir,
                    model_name=args.model,
                    repo_id=repo_id,
                    train_secs=args.train_secs,
                    topk=args.topk,
                    rank=args.rank,
                    grad_accum=low_accum,
                )
            except Exception as e2:
                logger.error("[%s] OOM retry also failed: %s", comp_key, e2)
                res = {
                    "model": _normalize_base_model_id(args.model),
                    "compressor": comp_key,
                    "status": "SKIPPED_OOM",
                    "error": str(e2)[:240],
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }

        if res is not None:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
                f.flush()

            if res.get("status") == "PASSED" and args.hf_upload and uploader:
                uploader.submit_upload(
                    repo_id=repo_id,
                    folder_path=str(ckpt_dir),
                    delete_after_upload=bool(not args.keep_local_weights),
                )

    if uploader:
        logger.info("Waiting for background HF uploads...")
        uploader.shutdown(wait=True)

    logger.info("Qwen Compressor Zoo run finished.")
    os._exit(0)


if __name__ == "__main__":
    main()
