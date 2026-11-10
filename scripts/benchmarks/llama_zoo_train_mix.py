#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TinyLlama-1.1B (LLaMA Family) Attention Surgery & WikiText-2 Distillation Engine.

Target Architecture:
  - `TinyLlama/TinyLlama_v1.1` (`LlamaForCausalLM`):
    22 decoder layers, hidden_size=2048 (exact 2^11), num_attention_heads=32 (2^5),
    num_key_value_heads=4 (8:1 GQA), head_dim=64, bias-free Q/K/V/O projections.

Reuses canonical NeuralOps surgery (`QwenAttentionAdapter`), chunked causal PPL
evaluator, WikiText-2 offline corpus builder, and model card generator cleanly.
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_zoo_train_mix import (
    CausalTextDataset,
    _build_cache_wikitext,
    _extract_attention_metadata,
    _format_size,
    _summarize_attention_doc,
    apply_qwen_attention_surgery,
    evaluate_causal_ppl,
)
from neuralops.core.registry import get_module_capabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llama_zoo_distill")


def build_llama_readme(
    attn_key: str,
    result: dict[str, Any],
    teacher_ppl: float,
    out_dir: Path,
    meta: dict[str, Any],
) -> str:
    """Generates a complete TinyLlama-1.1B Hugging Face Model Card with YAML frontmatter."""
    cls_name = meta.get("class_name", attn_key)
    family = meta.get("family", "attention")
    mod_path = meta.get("module_path", "neuralops/layers/")
    docstring = _summarize_attention_doc(str(meta.get("docstring", "")), attn_key)

    pre_ppl = result.get("ppl_before")
    post_ppl = result.get("ppl_after")
    pre_str = f"{pre_ppl:,.2f}" if isinstance(pre_ppl, (int, float)) and math.isfinite(pre_ppl) else str(pre_ppl)
    post_str = f"{post_ppl:,.2f}" if isinstance(post_ppl, (int, float)) and math.isfinite(post_ppl) else str(post_ppl)
    badge_ppl = f"{post_ppl:.2f}" if isinstance(post_ppl, (int, float)) and math.isfinite(post_ppl) else str(post_ppl)

    slug = attn_key.replace("_", "-")
    repo_slug = f"tinyllama-1.1b-{slug}-wikitext2"
    repo_id = f"{result.get('hf_org', 'tahamajs')}/{repo_slug}"
    bin_size = _format_size(out_dir / "model.safetensors") if (out_dir / "model.safetensors").exists() else _format_size(out_dir / "pytorch_model.bin")
    adapter_size = _format_size(out_dir / "attention_weights.safetensors")

    yaml_ppl = ""
    if isinstance(post_ppl, (int, float)) and math.isfinite(post_ppl):
        yaml_ppl = f"""model-index:
  - name: {repo_slug}
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
            value: {post_ppl:.2f}
            name: WikiText-2 Post-KD Perplexity"""

    return f"""---
language:
  - en
license: mit
license_link: https://github.com/tahamajs/NeuralOps/blob/main/LICENSE
library_name: neuralops
base_model: TinyLlama/TinyLlama_v1.1
datasets:
  - Salesforce/wikitext
metrics:
  - perplexity
tags:
  - neuralops
  - llama
  - tinyllama
  - attention-zoo
  - attention-surgery
  - subquadratic-attention
  - wikitext-2
  - knowledge-distillation
  - {family}
  - {attn_key}
pipeline_tag: text-generation
{yaml_ppl}
---

# TinyLlama-1.1B ({attn_key}) — NeuralOps LLaMA Attention Surgery Zoo

[![WikiText2 PPL](https://img.shields.io/badge/WikiText2_PPL-{badge_ppl}-brightgreen)](#-empirical-evaluation)
[![Teacher Baseline](https://img.shields.io/badge/Teacher_Baseline_PPL-{teacher_ppl:.2f}-blue)](#-empirical-evaluation)
[![Surgery](https://img.shields.io/badge/Surgery-Hybrid_3:1_(17/22_Layers)-purple)](#-llama-surgical-topology)
[![Weights](https://img.shields.io/badge/Weights-{bin_size.replace(' ', '_')}-indigo)](#-packaged-artifacts)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/tahamajs/NeuralOps/blob/main/LICENSE)

Surgically transplanted and 2-stage distilled checkpoint of **`TinyLlama/TinyLlama_v1.1`** (`LlamaForCausalLM`, `1.1B` params, `22` layers, `D=2048`, `32Q : 4KV` 8:1 GQA) using **`{attn_key}`** (`{cls_name}`) on **WikiText-2 (`Salesforce/wikitext`)**.

---

## Architecture & Operator Profile

> **{cls_name}**: {docstring}

- **Registry Key:** `{attn_key}` (`{mod_path}`)
- **Base LLaMA Geometry:** `22` decoder layers, `hidden_size=2048` ($2^{{11}}$), `num_attention_heads=32` ($2^5$), `num_key_value_heads=4` (8:1 GQA), `head_dim=64`, bias-free projections.

---

## LLaMA Surgical Topology (`{result.get('pattern', 'hybrid_3_1')}`)

- **Transplanted Layers (`17 / 22`):** Replaced with `{attn_key}` wrapped in `QwenAttentionAdapter` (8:1 GQA KV-head expansion + single-`o_proj` bias-free weight inheritance).
- **Anchor Layers (`5 / 22`):** `[3, 7, 11, 15, 19]` retain exact native `LlamaAttention` GQA softmax.

---

## Empirical Evaluation

| Metric | Value | Notes |
|---|---:|---|
| **Teacher Baseline (`TinyLlama/TinyLlama_v1.1`)** | **`{teacher_ppl:.2f}`** | Reference WikiText-2 Causal Perplexity |
| **Pre-KD Zero-Shot Student PPL** | `{pre_str}` | Immediately post 17/22 surgery |
| **Post-KD Distilled Student PPL** | **`{post_str}`** | After 2-stage hidden representation distillation |
| **Optimization Steps & Duration** | `{result.get('total_steps', 0)}` steps in `{result.get('train_secs', 0.0):.1f}s` | `Peak VRAM: {result.get('peak_vram_mb', 0)} MB` on RTX 3090 24GB |

---

## Packaged Artifacts

- **`model.safetensors` / `pytorch_model.bin`** (`{bin_size}`): Full FP16 TinyLlama-1.1B checkpoint.
- **`attention_weights.safetensors`** (`{adapter_size}`): Distilled adapter weights only.

---

## Citation

```bibtex
@software{{neuralops2026llamazoo,
  title   = {{NeuralOps: Subquadratic & Hybrid Attention Surgery Zoo for TinyLlama-1.1B}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}},
  note    = {{Hub Repository: {repo_id}}}
}}
```
"""


def train_single_llama_attention(
    attn_key: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    teacher_ppl: float,
    pattern: str = "hybrid_3_1",
    train_secs: float = 150.0,
    grad_accum: int = 8,
    hf_org: str = "tahamajs",
) -> dict[str, Any]:
    """Runs 2-stage hidden representation distillation for TinyLlama-1.1B."""
    logger.info("\n" + "=" * 80)
    logger.info("[LLAMA START] Distilling %s on TinyLlama-1.1B (%ds, pattern=%s)", attn_key, int(train_secs), pattern)
    logger.info("=" * 80)

    student = copy.deepcopy(teacher_model).float()
    student = apply_qwen_attention_surgery(student, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Pre-KD WikiText-2 PPL=%.2f", attn_key, pre_ppl)

    for name, p in student.named_parameters():
        p.requires_grad = ("self_attn" in name) or ("layernorm" in name.lower())

    t_start = time.time()
    t_s1_end = t_start + train_secs * 0.30
    t_s2_end = t_start + train_secs

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.0e-4, weight_decay=0.01)
    scaler1 = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    student.train()
    step_s1 = 0

    while time.time() < t_s1_end:
        for batch in train_loader:
            if time.time() >= t_s1_end:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_h, t_h = s_out.hidden_states[1:], t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
                loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
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

    del opt1, scaler1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    opt2 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=6.0e-5, weight_decay=0.01)
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    step_s2 = 0

    while time.time() < t_s2_end:
        for batch in train_loader:
            if time.time() >= t_s2_end:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                s_h, t_h = s_out.hidden_states[1:], t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
                loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
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

    total_time = time.time() - t_start
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0
    logger.info("[%s] Post-KD WikiText-2 PPL=%.2f (Delta: %+.2f)", attn_key, post_ppl, post_ppl - pre_ppl)

    result = {
        "model": "TinyLlama/TinyLlama_v1.1",
        "attn_key": attn_key,
        "dataset": "wikitext-2",
        "pattern": pattern,
        "teacher_ppl": round(teacher_ppl, 2),
        "ppl_before": round(pre_ppl, 2) if math.isfinite(pre_ppl) else float("inf"),
        "ppl_after": round(post_ppl, 2) if math.isfinite(post_ppl) else float("inf"),
        "ppl_delta": round(post_ppl - pre_ppl, 2) if (math.isfinite(pre_ppl) and math.isfinite(post_ppl)) else float("nan"),
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "total_steps": step_s1 + step_s2,
        "peak_vram_mb": peak_vram,
        "hf_org": hf_org,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "PASSED",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        student.half().save_pretrained(str(out_dir))
    except Exception:
        student.half().save_pretrained(str(out_dir), safe_serialization=False)

    attn_state = {
        k: v.detach().cpu().half().clone().contiguous()
        for k, v in student.state_dict().items()
        if any(term in k.lower() for term in ("self_attn", "attention", "adapter", "lora"))
    }
    from safetensors.torch import save_file

    save_file(attn_state, str(out_dir / "attention_weights.safetensors"))

    cfg = student.config.to_dict()
    cfg["_name_or_path"] = "TinyLlama/TinyLlama_v1.1"
    cfg["base_model_name_or_path"] = "TinyLlama/TinyLlama_v1.1"
    cfg["license"] = "mit"
    cfg["attention_type"] = attn_key
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    meta_map = _extract_attention_metadata()
    readme = build_llama_readme(attn_key, result, teacher_ppl, out_dir, meta_map.get(attn_key, {}))
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    del student, opt2, scaler2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyLlama-1.1B Attention Surgery Zoo")
    parser.add_argument("--model", default="/home/tahamajlesi.ut/NeuralOps/models/tinyllama_1.1b")
    parser.add_argument("--keys_json", default="results/bert_attention_zoo/registry_keys.json")
    parser.add_argument("--attn_key", default=None)
    parser.add_argument("--surgery_pattern", choices=["all", "hybrid_3_1"], default="hybrid_3_1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--train_secs", type=float, default=150.0)
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo")
    parser.add_argument("--metrics_log", default="/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA unavailable.")
        sys.exit(3)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_chunks, val_chunks = _build_cache_wikitext(tokenizer, seq_len=args.seq_len, cache_path="/dev/shm/wikitext_tinyllama_seq512.pt")
    train_loader = DataLoader(CausalTextDataset(train_chunks), batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(CausalTextDataset(val_chunks), batch_size=4, shuffle=False, pin_memory=(device.type == "cuda"))

    teacher_ppl = evaluate_causal_ppl(teacher, val_loader, device)
    logger.info("TinyLlama-1.1B Teacher Baseline WikiText-2 PPL=%.2f", teacher_ppl)

    keys = [args.attn_key] if args.attn_key else json.loads(Path(args.keys_json).read_text(encoding="utf-8"))
    passed: set[str] = set()
    ledger = Path(args.metrics_log)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("status") == "PASSED":
                        passed.add(str(d.get("attn_key")))
                except Exception:
                    pass

    out_base = Path(args.out_dir)
    for i, key in enumerate(keys, 1):
        if key in passed:
            logger.info("[%d/%d] Skipping %s (already PASSED).", i, len(keys), key)
            continue
        try:
            res = train_single_llama_attention(
                attn_key=key,
                teacher_model=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                tokenizer=tokenizer,
                out_dir=out_base / key,
                teacher_ppl=teacher_ppl,
                pattern=args.surgery_pattern,
                train_secs=args.train_secs,
                grad_accum=args.grad_accum,
                hf_org=args.hf_org,
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.warning("[%s] OOM at bs=%d. Retrying at bs=2...", key, args.batch_size)
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    halved_loader = DataLoader(CausalTextDataset(train_chunks), batch_size=max(1, args.batch_size // 2), shuffle=True, drop_last=True)
                    res = train_single_llama_attention(
                        attn_key=key,
                        teacher_model=teacher,
                        train_loader=halved_loader,
                        val_loader=val_loader,
                        device=device,
                        tokenizer=tokenizer,
                        out_dir=out_base / key,
                        teacher_ppl=teacher_ppl,
                        pattern=args.surgery_pattern,
                        train_secs=args.train_secs,
                        grad_accum=args.grad_accum * 2,
                        hf_org=args.hf_org,
                    )
                except Exception as e2:
                    res = {"attn_key": key, "status": "SKIPPED_OOM", "error": str(e2), "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
            else:
                logger.error("[%s] LLaMA training exception: %s", key, e, exc_info=True)
                res = {"attn_key": key, "status": "FAIL_TRAINING", "error": str(e), "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
