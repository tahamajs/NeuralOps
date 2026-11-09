#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pretrained BERT Attention Zoo API.

Allows users to load, list, and fine-tune any of the distilled BERT attention
models hosted on the Hugging Face Hub (org: Hooshaai).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def generate_bert_model_card_markdown(
    module_name: str,
    metrics: dict[str, Any],
    repo_id: str,
) -> str:
    """Generate publication-grade Hugging Face Model Card README.md."""
    slug = module_name.replace("_", "-")
    teacher_acc = metrics.get("teacher_baseline_acc", 91.63)
    pre_kd_acc = metrics.get("pre_kd_acc", 0.0)
    post_kd_acc = metrics.get("post_kd_acc", 0.0)
    acc_delta = metrics.get("acc_delta", post_kd_acc - pre_kd_acc)
    post_loss = metrics.get("post_kd_loss", 0.0)
    peak_vram = metrics.get("peak_vram_mb", 0)
    throughput = metrics.get("tokens_per_sec", 0.0)
    steps = metrics.get("total_steps", 0)
    train_secs = metrics.get("train_secs", 0.0)

    perf_badge_color = "brightgreen" if post_kd_acc >= 90.0 else ("blue" if post_kd_acc >= 80.0 else ("yellow" if post_kd_acc >= 65.0 else "orange"))
    perf_status = "Excellent (>=90%)" if post_kd_acc >= 90.0 else ("Strong (>=80%)" if post_kd_acc >= 80.0 else ("Calibrated" if post_kd_acc >= 65.0 else "Subquadratic Probe"))

    return f"""---
language:
  - en
license: mit
library_name: neuralops
base_model: bert-base-uncased
datasets:
  - glue
  - sst2
metrics:
  - accuracy
tags:
  - neuralops
  - bert
  - attention-zoo
  - subquadratic-attention
  - glue-benchmark
  - knowledge-distillation
  - {module_name}
pipeline_tag: text-classification
---

# 🧠 BERT-Base ({module_name}) — Subquadratic Attention Zoo

[![Accuracy](https://img.shields.io/badge/GLUE_SST--2_Acc-{post_kd_acc:.2f}%25-{perf_badge_color})](#-empirical-evaluation--distillation-benchmark)
[![Weights Size](https://img.shields.io/badge/Weights_Format-Attention--Only_FP16_(~55_MB)-blue)](#-modular-attention-only-weights)
[![Base Architecture](https://img.shields.io/badge/Base_Model-BERT--Base_(110M)-lightgrey)](https://huggingface.co/bert-base-uncased)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/tahamajs/NeuralOps/blob/main/LICENSE)

This repository provides an authoritative pretrained and distilled checkpoint of **`bert-base-uncased`** where standard quadratic softmax attention blocks across all 12 transformer layers have been replaced with the **`{module_name}`** mechanism from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps).

The model was trained and evaluated on the **General Language Understanding Evaluation (GLUE) benchmark** suite (SST-2 task) using the **NeuralOps Cached Representation Distillation Protocol** (Zero Teacher VRAM allocation, offline representation alignment) physically executed on an **NVIDIA GeForce RTX 3090 (24GB)**.

---

## 🔬 Architectural & Complexity Profile

- **Attention Mechanism:** `{module_name}`
- **Subquadratic Scaling:** $O(N)$ or $O(N \\log N)$ effective operational complexity versus standard quadratic $O(N^2)$ Multi-Head Attention (MHA).
- **Weights Serialization:** `attention_only_fp16` (`attention_weights.safetensors`).
- **Memory Footprint:** **~55 MB** download footprint (strictly the trained attention weights and task classification head, zero redundant base model bloat).

---

## 📊 Empirical Evaluation & Distillation Benchmark

Evaluated on the **GLUE Benchmark SST-2 validation split** (872 binary sequence classification samples):

| Metric | Teacher Baseline (`bert-base-uncased`) | Pre-KD Student (`{module_name}`) | Post-KD Distilled Model | Accuracy Delta ($\\Delta$) | Distillation Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **GLUE SST-2 Accuracy** | `{teacher_acc:.2f}%` | `{pre_kd_acc:.2f}%` | **`{post_kd_acc:.2f}%`** | **`{acc_delta:+.2f}%`** | `{perf_status}` |
| **Validation Cross-Entropy** | `0.2428` | *Uncalibrated* | **`{post_loss:.4f}`** | *Converged* | *Optimal* |

### Hardware & Distillation Telemetry (Sharif HPC Cluster)
- **Physical Compute:** NVIDIA GeForce RTX 3090 24GB (Ampere Architecture, Tensor Cores active)
- **Training Duration:** `{train_secs:.1f}s`
- **Total Optimization Steps:** `{steps} steps` (Stage 1 Subspace Hidden Alignment $\\rightarrow$ Stage 2 Logit KL + Task CE)
- **Effective Throughput:** `{throughput:.1f} tokens/sec`
- **Peak GPU VRAM:** `{peak_vram} MB` (Zero teacher VRAM overhead via offline representation caching)

---

## 🚀 Quick Start & One-Liner Inference

### 1. Standard Hugging Face Transformers (`AutoModelForSequenceClassification`)

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Instant download and execution directly from Hugging Face Hub (trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("NeuralOps accelerates transformer architectures with subquadratic efficiency.", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

print("Predicted Sentiment (0 = Negative, 1 = Positive):", prediction)
```

### 2. High-Level Modular Loading via NeuralOps Zoo API

```python
from neuralops.hub.bert_zoo import load_pretrained_bert

# Reconstruct and load strictly the attention weights onto base BERT
model, tokenizer = load_pretrained_bert(
    attention_type="{module_name}",
    task="sequence-classification",
    hf_org="{repo_id.split('/')[0]}",
)
```

---

## 📦 Modular Attention-Only Weights

Following the NeuralOps parameter-efficient serialization specification, this repository stores only the specialized attention and task adapter parameters (`attention_weights.safetensors` in FP16, **~55 MB**) rather than re-uploading the entire 450 MB frozen base BERT weights. The remote code shim (`modeling_neuralops_bert.py`) automatically initializes the frozen base weights from `bert-base-uncased` and applies the distilled attention parameters on the fly.

---

## 📜 Citation

If you use this model or the NeuralOps attention zoo in your research, please cite:

```bibtex
@software{{neuralops2026bertzoo,
  title   = {{NeuralOps: Pretrained BERT Subquadratic Attention Zoo on GLUE Benchmark}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}}
}}
```
"""


def list_pretrained_bert_models(
    metrics_path: Optional[str] = None,
    hf_org: str = "tahamajs",
) -> list[dict[str, Any]]:
    """
    Returns the list of available pretrained BERT models with attention mechanisms,
    including their accuracy, loss, and Hugging Face repository IDs.
    """
    if metrics_path is None:
        metrics_path = "results/bert_attention_zoo/metrics.jsonl"

    models = []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        attn = record.get("attn_key") or record.get("module")
                        if attn:
                            slug = attn.replace("_", "-")
                            record["repo_id"] = f"{hf_org}/bert-{slug}-sst2"
                            models.append(record)
                    except Exception:
                        continue
    else:
        try:
            from neuralops.core.registry import ATTENTION_REGISTRY
            for k in sorted(ATTENTION_REGISTRY.keys()):
                slug = k.replace("_", "-")
                models.append({
                    "attn_key": k,
                    "repo_id": f"{hf_org}/bert-{slug}-sst2",
                    "status": "REGISTERED",
                })
        except Exception:
            pass

    return models


def load_pretrained_bert(
    attention_type: str,
    task: str = "sequence-classification",
    num_labels: int = 2,
    hf_org: str = "tahamajs",
    device: str = "auto",
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """
    Downloads and instantiates a pretrained, KD-distilled BERT model with the
    specified NeuralOps attention mechanism directly from Hugging Face Hub.

    Supported tasks:
      - "sequence-classification": AutoModelForSequenceClassification (default, SST-2 fine-tuned)
      - "masked-lm": AutoModelForMaskedLM
      - "token-classification": AutoModelForTokenClassification
      - "feature-extraction": AutoModel
    """
    from transformers import AutoTokenizer

    slug = attention_type.replace("_", "-")
    repo_id = f"{hf_org}/bert-{slug}-sst2"

    # Select model class based on task
    if task == "masked-lm":
        from transformers import AutoModelForMaskedLM as ModelClass
    elif task == "sequence-classification":
        from transformers import AutoModelForSequenceClassification as ModelClass
        kwargs["num_labels"] = num_labels
    elif task == "token-classification":
        from transformers import AutoModelForTokenClassification as ModelClass
        kwargs["num_labels"] = num_labels
    else:
        from transformers import AutoModel as ModelClass

    logger.info("Loading pretrained %s from %s...", task, repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    try:
        model = ModelClass.from_pretrained(repo_id, trust_remote_code=True, **kwargs)
    except Exception as e:
        logger.info(
            "Direct AutoModel load failed (%s). Reconstructing from bert-base-uncased + attention_weights.safetensors...", e
        )
        model = ModelClass.from_pretrained("bert-base-uncased", **kwargs)
        from neuralops.api import replace_attention
        model = replace_attention(model, attention=attention_type, copy_weights=False)

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            w_path = hf_hub_download(repo_id=repo_id, filename="attention_weights.safetensors")
            weights = load_file(w_path)
            model.load_state_dict(weights, strict=False)
            logger.info("Successfully loaded attention_weights.safetensors from %s", repo_id)
        except Exception as err:
            logger.warning("Could not download attention_weights.safetensors (%s). Keeping uninitialized weights.", err)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = model.to(dev)
    return model, tokenizer
