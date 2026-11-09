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
    teacher_ppl = metrics.get("teacher_ppl", 6.84)
    teacher_acc = metrics.get("teacher_acc", 68.42)
    raw_ppl = metrics.get("raw_uncalibrated_ppl", metrics.get("ppl_before", 0.0))
    raw_acc = metrics.get("raw_uncalibrated_acc", metrics.get("acc_before", 0.0))
    pre_kd_ppl = metrics.get("pre_kd_ppl", metrics.get("ppl_before", 0.0))
    pre_kd_acc = metrics.get("pre_kd_acc", metrics.get("acc_before", 0.0))
    post_kd_ppl = metrics.get("post_kd_ppl", metrics.get("ppl_after", 0.0))
    post_kd_acc = metrics.get("post_kd_acc", metrics.get("acc_after", 0.0))
    peak_vram = metrics.get("peak_vram_mb", 0)
    throughput = metrics.get("tokens_per_sec", 0.0)
    steps = metrics.get("total_steps", metrics.get("steps_completed", 0))

    pre_delta = pre_kd_ppl - teacher_ppl
    post_delta = post_kd_ppl - teacher_ppl

    return f"""---
language:
  - en
license: mit
library_name: neuralops
base_model: bert-base-uncased
datasets:
  - wikitext
metrics:
  - perplexity
  - accuracy
tags:
  - neuralops
  - bert
  - attention-zoo
  - knowledge-distillation
  - subquadratic-attention
  - {module_name}
pipeline_tag: fill-mask
---

# 🧠 BERT-Base ({module_name}) — Distilled on WikiText-2

This model is a pretrained variant of **`bert-base-uncased`** where all 12 standard Multi-Head Self-Attention layers have been replaced with the **`{module_name}`** mechanism from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps).

The model was distilled on **WikiText-2** using the **NeuralOps 3-Stage GPU-Saturated Knowledge Distillation Pipeline** for 300 seconds on an NVIDIA RTX 3090.

---

## 📊 Empirical Evaluation & Distillation Benchmark

Evaluated using **Deterministic Masked Language Modeling Pseudo-Perplexity ($\text{{MLM-PPL}}$)** on WikiText-2 (15% token masking, `seed=42`):

| Evaluation Phase | MLM Perplexity ($\text{{PPL}}$) | Masked Token Acc (%) | Delta vs Baseline |
|---|:---:|:---:|:---:|
| **Vanilla BERT Teacher (`bert-base-uncased`)** | `{teacher_ppl:.2f}` | `{teacher_acc:.2f}%` | *Reference* |
| **Stage 0 (Closed-Form Ridge Calibration)** | `{pre_kd_ppl:.2f}` | `{pre_kd_acc:.2f}%` | `{pre_delta:+.2f}` |
| **Stage 2 (Post-KD Distilled Model)** | **`{post_kd_ppl:.2f}`** | **`{post_kd_acc:.2f}%`** | **`{post_delta:+.2f}`** |

### Hardware & Training Telemetry
- **Distillation Hardware:** NVIDIA RTX 3090 24GB (Sharif HPC Cluster)
- **Training Duration:** `300.0s (5.0 minutes)`
- **Steps Completed:** `{steps}`
- **Throughput:** `{throughput:.1f} tokens/sec`
- **Peak GPU VRAM:** `{peak_vram} MB`
- **Distillation Curriculum:** Stage 0 (Closed-Form Ridge) $\\rightarrow$ Stage 1 (Frozen-Backbone Subspace MSE/CKA) $\\rightarrow$ Stage 2 (Differential-LR Fused Logit KD)

---

## 🚀 Quick Start & Usage

### 1. Using NeuralOps High-Level API (Recommended)

```python
import neuralops

# Instant load from Hugging Face Hub
model, tokenizer = neuralops.load_pretrained_bert(
    attention_type="{module_name}",
    task="masked-lm",
)

text = "NeuralOps makes attention mechanisms [MASK] and efficient."
inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits
```

### 2. Standard Transformers Pipeline

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="{repo_id}")
results = unmasker("NeuralOps makes deep learning models [MASK] and efficient.")
for res in results:
    print(f"{{res['token_str']}}: {{res['score']:.4f}}")
```

---

## 📜 Citation

```bibtex
@software{{neuralops2026bertzoo,
  title   = {{NeuralOps: Pretrained BERT-131 Subquadratic Attention Zoo}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}}
}}
```
"""


def list_pretrained_bert_models(
    metrics_path: Optional[str] = None,
    hf_org: str = "Hooshaai",
) -> list[dict[str, Any]]:
    """
    Returns the list of available pretrained BERT models with attention mechanisms,
    including their PPL, accuracy, and Hugging Face repository IDs.
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
                            record["repo_id"] = f"{hf_org}/bert-{slug}-wiki"
                            models.append(record)
                    except Exception:
                        continue
    else:
        # Fallback to loading registered keys
        try:
            from neuralops.core.registry import ATTENTION_REGISTRY
            for k in sorted(ATTENTION_REGISTRY.keys()):
                slug = k.replace("_", "-")
                models.append({
                    "attn_key": k,
                    "repo_id": f"{hf_org}/bert-{slug}-wiki",
                    "status": "REGISTERED",
                })
        except Exception:
            pass

    return models


def load_pretrained_bert(
    attention_type: str,
    task: str = "masked-lm",
    num_labels: int = 2,
    hf_org: str = "Hooshaai",
    device: str = "auto",
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """
    Downloads and instantiates a pretrained, KD-distilled BERT model with the
    specified NeuralOps attention mechanism directly from Hugging Face Hub.

    Supported tasks:
      - "masked-lm": AutoModelForMaskedLM
      - "sequence-classification": AutoModelForSequenceClassification
      - "token-classification": AutoModelForTokenClassification
      - "feature-extraction": AutoModel
    """
    from transformers import AutoTokenizer

    slug = attention_type.replace("_", "-")
    repo_id = f"{hf_org}/bert-{slug}-wiki"

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
        logger.warning(
            "Remote hub download failed (%s). Falling back to local surgery on bert-base-uncased.", e
        )
        model = ModelClass.from_pretrained("bert-base-uncased", **kwargs)
        from neuralops.api import replace_attention
        model = replace_attention(model, attention=attention_type, copy_weights=True)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = model.to(dev)
    return model, tokenizer
