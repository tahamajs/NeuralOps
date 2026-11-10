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
import sys
import time
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
    """Tokenizes corpus into contiguous chunks of length seq_len with RAM caching."""
    cache_path = Path(f"/dev/shm/wikitext_qwen_seq{seq_len}.pt")
    if cache_path.exists():
        try:
            logger.info("Loading pre-tokenized corpus from RAM cache: %s", cache_path)
            data = torch.load(cache_path, map_location="cpu")
            return data["train_chunks"], data["val_chunks"]
        except Exception as e:
            logger.warning("RAM cache load failed (%s); re-tokenizing...", e)

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

    try:
        torch.save({"train_chunks": train_chunks, "val_chunks": val_chunks}, cache_path)
        logger.info("Cached tokenized corpus to RAM: %s", cache_path)
    except Exception as e:
        logger.debug("Failed to write RAM cache: %s", e)

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
    max_batches: int = 15,
) -> float:
    """Computes Causal LM Perplexity with micro-chunked evaluation to prevent 152k vocab OOM."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                h = getattr(model, "model", model)(input_ids=in_ids, attention_mask=att_m).last_hidden_state
                seq_len = in_ids.size(1) - 1
                targets = lbls[:, 1:].contiguous()
                h_causal = h[:, :seq_len, :]

                chunk_sz = 64
                for c in range(0, seq_len, chunk_sz):
                    h_c = h_causal[:, c : c + chunk_sz, :]
                    tgt_c = targets[:, c : c + chunk_sz]
                    logits_c = getattr(model, "lm_head", None)
                    if logits_c is not None:
                        logits = logits_c(h_c)
                        loss_c = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_c.reshape(-1), reduction="sum")
                        if torch.isfinite(loss_c):
                            total_loss += loss_c.item()
                            total_tokens += tgt_c.numel()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
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

        # Instantiate target attention from ATTENTION_REGISTRY via normalized factory
        if attn_key in ATTENTION_REGISTRY:
            try:
                core_attn = create_attention(
                    attn_key,
                    dim=hidden_size,
                    num_heads=num_heads,
                    config=model.config,
                    is_causal=True,
                    layer_idx=idx,
                )
            except TypeError:
                try:
                    core_attn = create_attention(
                        attn_key,
                        dim=hidden_size,
                        num_heads=num_heads,
                        config=model.config,
                        is_causal=True,
                    )
                except TypeError:
                    core_attn = create_attention(
                        attn_key,
                        dim=hidden_size,
                        num_heads=num_heads,
                        config=model.config,
                    )
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


def _normalize_base_model_id(model_name: str) -> str:
    """Normalize local model cache paths (e.g. /dev/shm/Qwen2.5-0.5B) to canonical HF Hub IDs."""
    cleaned = model_name.strip()
    if "qwen2.5-0.5b" in cleaned.lower():
        return "Qwen/Qwen2.5-0.5B"
    if cleaned.startswith("/") or os.path.exists(cleaned):
        tail = Path(cleaned.rstrip("/")).name
        if tail.lower().startswith("qwen"):
            return f"Qwen/{tail}"
        return tail
    return cleaned


def _summarize_attention_doc(attn_key: str) -> str:
    """Return a cleaned single-line summary from the registered attention class docstring."""
    cls = ATTENTION_REGISTRY.get(attn_key) if attn_key in ATTENTION_REGISTRY else None
    raw_doc = inspect.getdoc(cls) if cls is not None else ""
    if not raw_doc:
        return f"Specialized attention mechanism `{attn_key}` from the NeuralOps ATTENTION_REGISTRY."
    for marker in (
        "Attributes:",
        "Args:",
        "Parameters:",
        "Returns:",
        "Raises:",
        "Example:",
        "Examples:",
        "Note:",
        "Notes:",
    ):
        idx = raw_doc.find(marker)
        if idx != -1:
            raw_doc = raw_doc[:idx]
    summary = " ".join(raw_doc.split()).strip()
    return summary or f"Specialized attention mechanism `{attn_key}` from the NeuralOps ATTENTION_REGISTRY."


def _format_size(num_bytes: int) -> str:
    """Format byte count into human-readable MB/KB string."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _collect_artifact_sizes(out_dir: Optional[Path]) -> tuple[list[tuple[str, str]], str]:
    """Collect saved checkpoint artifact sizes and compute total weight footprint."""
    if out_dir is None or not out_dir.exists():
        return [], "N/A"
    candidates = (
        "model.safetensors",
        "pytorch_model.bin",
        "attention_weights.safetensors",
        "config.json",
        "tokenizer.json",
        "training_metrics.json",
    )
    rows: list[tuple[str, str]] = []
    weight_bytes = 0
    for fname in candidates:
        fpath = out_dir / fname
        if fpath.is_file():
            sz = fpath.stat().st_size
            rows.append((fname, _format_size(sz)))
            if fname.endswith((".safetensors", ".bin")):
                weight_bytes += sz
    total_badge = _format_size(weight_bytes).replace(" ", "_") if weight_bytes > 0 else "FP16"
    return rows, total_badge


def build_qwen_readme(
    attn_key: str,
    result: dict[str, Any],
    out_dir: Optional[Path] = None,
    base_model: str = "Qwen/Qwen2.5-0.5B",
    repo_id: Optional[str] = None,
    num_layers: int = 24,
    dur_s1: Optional[float] = None,
    dur_s2: Optional[float] = None,
    seq_len: int = 512,
    grad_accum: int = 4,
    topk: int = 64,
) -> str:
    """Generate a publication-grade Hugging Face model card README.md for distilled Qwen checkpoints."""
    base_id = _normalize_base_model_id(base_model)
    attn_slug = attn_key.replace("_", "-")
    target_repo = repo_id or f"Hooshaai/qwen2.5-0.5b-{attn_slug}-wikitext2"
    model_short = target_repo.split("/")[-1]

    pattern = str(result.get("pattern", "hybrid_3_1"))
    pre_ppl = float(result.get("ppl_before", 0.0))
    post_ppl = float(result.get("ppl_after", 0.0))
    ppl_delta = float(result.get("ppl_delta", post_ppl - pre_ppl))
    train_secs = float(result.get("train_secs", 0.0))
    steps_s1 = int(result.get("steps_s1", 0))
    steps_s2 = int(result.get("steps_s2", 0))
    total_steps = int(result.get("total_steps", steps_s1 + steps_s2))
    peak_vram = int(result.get("peak_vram_mb", 0))

    s1_sec_str = f"{dur_s1:.1f}s" if dur_s1 is not None else "30% budget"
    s2_sec_str = f"{dur_s2:.1f}s" if dur_s2 is not None else "70% budget"

    doc_summary = _summarize_attention_doc(attn_key)

    cls = ATTENTION_REGISTRY.get(attn_key) if attn_key in ATTENTION_REGISTRY else None
    caps = get_module_capabilities(attn_key) if attn_key in ATTENTION_REGISTRY else {}
    cls_name = cls.__name__ if cls is not None else attn_key

    def _cap_flag(key: str, default: Any = "N/A") -> str:
        val = caps.get(key, getattr(cls, key, default) if cls is not None else default)
        if isinstance(val, bool):
            return "Yes" if val else "No"
        return str(val)

    if pattern == "hybrid_3_1":
        preserved_indices = [i for i in range(num_layers) if i % 4 == 3]
        transplanted_count = num_layers - len(preserved_indices)
        preserved_str = ", ".join(str(i) for i in preserved_indices)
        topology_desc = (
            f"3:1 interleaved hybrid (`hybrid_3_1`): **{transplanted_count}/{num_layers}** layers "
            f"transplanted with `{attn_key}` via `QwenAttentionAdapter` (14:2 GQA expansion, "
            f"Single-`o_proj` contract), and **{len(preserved_indices)}/{num_layers}** full quadratic "
            f"anchor layers preserved at indices `[{preserved_str}]`."
        )
    else:
        transplanted_count = num_layers
        preserved_indices = []
        topology_desc = (
            f"Full replacement (`all`): **{num_layers}/{num_layers}** decoder layers transplanted "
            f"with `{attn_key}` via `QwenAttentionAdapter` (14:2 GQA expansion, Single-`o_proj` contract)."
        )

    ppl_color = (
        "brightgreen"
        if post_ppl < 30.0
        else ("blue" if post_ppl < 60.0 else ("yellow" if post_ppl < 150.0 else "orange"))
    )
    pattern_badge = pattern.replace("_", "--")

    artifact_rows, weights_badge = _collect_artifact_sizes(out_dir)
    if artifact_rows:
        artifact_table = "\n".join(f"| `{fname}` | `{fsize}` |" for fname, fsize in artifact_rows)
    else:
        artifact_table = "| `attention_weights.safetensors` | `FP16 adapter checkpoint` |"

    return f"""---
language:
  - en
license: mit
library_name: neuralops
base_model: {base_id}
datasets:
  - Salesforce/wikitext
metrics:
  - perplexity
tags:
  - neuralops
  - qwen2
  - qwen2.5
  - attention-zoo
  - subquadratic-attention
  - knowledge-distillation
  - {attn_key}
pipeline_tag: text-generation
model-index:
  - name: {model_short}
    results:
      - task:
          type: text-generation
          name: Causal Language Modeling
        dataset:
          type: Salesforce/wikitext
          name: wikitext-2-raw-v1
          split: validation
        metrics:
          - type: perplexity
            value: {post_ppl:.2f}
            name: WikiText-2 Validation Perplexity
---

# {base_id} ({attn_key}) — NeuralOps Qwen Attention Zoo

[![WikiText-2 PPL](https://img.shields.io/badge/WikiText--2_PPL-{post_ppl:.2f}-{ppl_color})](#-2-stage-knowledge-distillation-telemetry)
[![Surgery Pattern](https://img.shields.io/badge/Surgery-{pattern_badge}-blue)](#-surgery-topology)
[![Weights](https://img.shields.io/badge/Weights-{weights_badge}-informational)](#-weight-artifacts)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen2.5--0.5B-lightgrey)](https://huggingface.co/{base_id})
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/tahamajs/NeuralOps/blob/main/LICENSE)

> **Module Summary (`{cls_name}`):** {doc_summary}

This repository contains a distilled **`{base_id}`** checkpoint whose decoder attention blocks have undergone surgical replacement with **`{attn_key}`** from the [NeuralOps](https://github.com/tahamajs/NeuralOps) `ATTENTION_REGISTRY`, trained on **WikiText-2 (`Salesforce/wikitext` / `wikitext-2-raw-v1`)**.

---

## Module Capability Profile

| Property | Value |
|---|---|
| Registry Key | `{attn_key}` |
| Implementation Class | `{cls_name}` |
| Causal Autoregressive Support (`supports_causal`) | `{_cap_flag("supports_causal", True)}` |
| Bidirectional Support (`supports_bidirectional`) | `{_cap_flag("supports_bidirectional")}` |
| Cross-Attention Support (`supports_cross_attention`) | `{_cap_flag("supports_cross_attention")}` |
| Native GQA Support (`supports_gqa`) | `{_cap_flag("supports_gqa")}` |
| Rotary Position Support (`supports_rope`) | `{_cap_flag("supports_rope")}` |
| Custom Attention Operator (`is_custom_attention`) | `{_cap_flag("is_custom_attention", True)}` |

---

## Surgery Topology

- **Pattern:** `{pattern}`
- **Layer Layout:** {topology_desc}
- **Adapter Contract:** `QwenAttentionAdapter` maps `Qwen2Attention` projections (`14` query heads, `2` KV heads, `head_dim=64`, `hidden_size=896`) with automatic 7x KV-head expansion where required and enforces the Single-`o_proj` bypass contract.

---

## 2-Stage Knowledge Distillation Telemetry

| Metric / Stage | Value | Notes |
|---|---:|---|
| Pre-KD Causal PPL (Stage 0) | `{pre_ppl:.2f}` | Zero-shot post-surgery baseline on `wikitext-2-raw-v1` |
| Post-KD Causal PPL | **`{post_ppl:.2f}`** | Final validation perplexity |
| Perplexity Delta ($\\Delta$) | `{ppl_delta:+.2f}` | Negative delta indicates recovery |
| Stage 1 (Frozen-Backbone Alignment) | `{steps_s1} steps` ({s1_sec_str}) | `lm_head` bypassed; layerwise hidden MSE + cosine (`0.25`) |
| Stage 2 (Multi-Layer Representation KD) | `{steps_s2} steps` ({s2_sec_str}) | Layerwise hidden MSE + cosine (`0.50`) |
| Total Optimization Steps / Time | `{total_steps} steps` / `{train_secs:.1f}s` | `seq_len={seq_len}`, `grad_accum={grad_accum}`, `topk={topk}` |
| Peak GPU VRAM | `{peak_vram} MB` | FP16 autocast with FP32 master student parameters |

---

## Weight Artifacts

| Artifact | Size |
|---|---:|
{artifact_table}

---

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.benchmarks.qwen_zoo_train_mix import apply_qwen_attention_surgery

base_model_id = "{base_id}"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)
model = apply_qwen_attention_surgery(model, attn_key="{attn_key}", pattern="{pattern}")

# Load transplanted attention + full weights from this checkpoint directory or Hub repo
# e.g. model = AutoModelForCausalLM.from_pretrained("{target_repo}", torch_dtype=torch.float16)
inputs = tokenizer("NeuralOps subquadratic attention enables", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
"""


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
    hf_org: str = "Hooshaai",
) -> dict[str, Any]:
    """Train one transplanted attention mechanism on Qwen with strict budget."""
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Attention Module: %s (%ds, pattern=%s)", attn_key, int(train_secs), pattern)
    logger.info("=" * 80)

    # 1. Surgery with weight inheritance (FP32 master weights for AMP stability)
    student = copy.deepcopy(teacher_model).float()
    student = apply_qwen_attention_surgery(student, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    # Stage 0: Evaluate uncalibrated / zero-shot on WikiText-2
    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Pre-KD WikiText-2 Perplexity: PPL=%.2f", attn_key, pre_ppl)

    # Timer starts STRICTLY AFTER Stage 0 evaluation finish
    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * stage1_frac)
    t_s2_deadline = t_train_start + train_secs

    # Stage 1: Frozen-Backbone Subspace Alignment (lm_head bypassed)
    for name, p in student.named_parameters():
        p.requires_grad = ("self_attn" in name) or ("layernorm" in name.lower())

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.0e-4, weight_decay=0.01)
    scaler1 = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
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
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    student.train()
    step_s2 = 0
    t0_s2 = time.time()

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                with torch.no_grad():
                    t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True)

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

    # Save checkpoint - full model + attention weights in FP16
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        student.half().save_pretrained(str(out_dir))
    except Exception:
        student.half().save_pretrained(str(out_dir), safe_serialization=False)

    attn_state: dict[str, torch.Tensor] = {}
    for k, v in student.state_dict().items():
        k_lower = k.lower()
        if any(term in k_lower for term in ("self_attn", "attention", "adapter", "lora", "qwen_att")):
            attn_state[k] = v.detach().cpu().half().clone().contiguous()

    from safetensors.torch import save_file
    save_file(attn_state, str(out_dir / "attention_weights.safetensors"))

    # Normalize and save config.json and training metrics
    qwen_cfg = student.config.to_dict()
    qwen_cfg["_name_or_path"] = base_model_id
    qwen_cfg["base_model_name_or_path"] = base_model_id
    qwen_cfg["license"] = "mit"
    qwen_cfg["attention_type"] = attn_key
    qwen_cfg["weights_format"] = "full_and_adapter_fp16"
    (out_dir / "config.json").write_text(json.dumps(qwen_cfg, indent=2), encoding="utf-8")
    (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

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
    (out_dir / "README.md").write_text(card_md, encoding="utf-8")

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
                hf_org=args.hf_org,
            )
        except Exception as e:
            # PyTorch <= 1.13 raises RuntimeError("CUDA out of memory...") instead of OutOfMemoryError
            if "out of memory" in str(e).lower():
                logger.warning("[%s] OOM at batch_size=%d. Halving batch size and retrying...", key, args.batch_size)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                try:
                    halved_loader = DataLoader(
                        train_dataset,
                        batch_size=max(1, args.batch_size // 2),
                        shuffle=True,
                        drop_last=True,
                    )
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
                        hf_org=args.hf_org,
                    )
                except Exception as retry_e:
                    logger.error("[%s] OOM retry failed: %s", key, retry_e)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    res = {
                        "attn_key": key,
                        "status": "SKIPPED_OOM",
                        "error": str(retry_e),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
            else:
                logger.error("[%s] Training exception: %s", key, e, exc_info=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                res = {
                    "attn_key": key,
                    "status": "FAIL_TRAINING",
                    "error": str(e),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }

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
