#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pretrained BERT & Generative Compressor Zoo API.

Allows users to load, list, and fine-tune any of the compressed and distilled
transformer models hosted on the Hugging Face Hub (org: Hooshaai).
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


def generate_compressor_model_card_markdown(
    compressor_name: str,
    metrics: dict[str, Any],
    repo_id: str,
    base_model: str = "bert-base-uncased",
) -> str:
    """Generate publication-grade Hugging Face Model Card README.md for compressed models."""
    slug = compressor_name.replace("_", "-")
    teacher_ppl = metrics.get("teacher_ppl", 10.48)
    teacher_acc = metrics.get("teacher_acc", 59.92)
    zero_shot_ppl = metrics.get("zero_shot_ppl", metrics.get("ppl_before", 0.0))
    zero_shot_acc = metrics.get("zero_shot_acc", metrics.get("acc_before", 0.0))
    post_kd_ppl = metrics.get("post_kd_ppl", metrics.get("ppl_after", 0.0))
    post_kd_acc = metrics.get("post_kd_acc", metrics.get("acc_after", 0.0))
    orig_params = metrics.get("orig_params", 109514298)
    comp_params = metrics.get("comp_params", orig_params)
    param_reduction = metrics.get("param_reduction", orig_params / max(comp_params, 1))
    reduction_pct = metrics.get("reduction_pct", (1.0 - comp_params / max(orig_params, 1)) * 100.0)
    throughput = metrics.get("tokens_per_sec", 0.0)
    peak_vram = metrics.get("peak_vram_mb", 0)
    steps = metrics.get("total_steps", metrics.get("steps_completed", 0))
    train_secs = metrics.get("train_secs", metrics.get("elapsed_secs", None))
    duration_str = f"{float(train_secs):.1f}s" if train_secs is not None else "Measured on RTX 3090"

    # Downstream task metrics if available (e.g. GLUE SST-2)
    teacher_sst2 = metrics.get("teacher_sst2_acc", 92.43)
    pre_kd_sst2 = metrics.get("pre_kd_sst2_acc", metrics.get("zero_shot_sst2_acc", None))
    post_kd_sst2 = metrics.get("post_kd_sst2_acc", None)

    sst2_section = ""
    if post_kd_sst2 is not None:
        pre_str = f"{pre_kd_sst2:.2f}%" if pre_kd_sst2 is not None else "N/A"
        sst2_section = f"""
### Downstream Benchmark: GLUE SST-2 Sequence Classification
| Model Variant | SST-2 Accuracy (%) | Compression Ratio | Status |
|---|:---:|:---:|:---:|
| **Teacher Baseline (`{base_model}`)** | `{teacher_sst2:.2f}%` | `1.00x` | *Full Accuracy Reference* |
| **Zero-Shot Compressed** | `{pre_str}` | `{param_reduction:.2f}x` | Post-Compression Drop |
| **Post-KD Restored Model** | **`{post_kd_sst2:.2f}%`** | **`{param_reduction:.2f}x`** | **Restored via 6-Pillar KD** |
"""

    return f"""---
language:
  - en
license: mit
library_name: neuralops
base_model: {base_model}
datasets:
  - wikitext
  - glue
metrics:
  - perplexity
  - accuracy
  - compression_ratio
tags:
  - neuralops
  - bert
  - compression-zoo
  - knowledge-distillation
  - model-compression
  - {compressor_name}
pipeline_tag: fill-mask
---

# 📦 BERT-Base Compressed ({compressor_name}) — Distilled on WikiText-2 & GLUE

This model is a compressed variant of **`{base_model}`** generated using the **`{compressor_name}`** compression engine from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps).

The model achieves **{param_reduction:.2f}x parameter reduction ({reduction_pct:.1f}% reduction)** and was distilled using the **NeuralOps 2-Stage Knowledge Distillation Pipeline** on an NVIDIA RTX 3090 GPU (Sharif HPC Cluster).

---

## 📊 Empirical Evaluation & Compression Telemetry

### Parameter & Compression Statistics
- **Original Model Parameters:** `{orig_params:,}`
- **Compressed Model Parameters:** `{comp_params:,}`
- **Parameters Saved:** `{orig_params - comp_params:,}`
- **Parameter Reduction Ratio:** **`{param_reduction:.2f}x` ({reduction_pct:.1f}% reduction)**

### Language Modeling Benchmark: WikiText-2 (Deterministic 15% MLM)
| Evaluation Phase | MLM Perplexity ($\text{{PPL}}$) | Masked Token Acc (%) | Delta vs Baseline |
|---|:---:|:---:|:---:|
| **Vanilla BERT Teacher (`{base_model}`)** | `{teacher_ppl:.2f}` | `{teacher_acc:.2f}%` | *Reference* |
| **Zero-Shot Compressed** | `{zero_shot_ppl:.2f}` | `{zero_shot_acc:.2f}%` | `{(zero_shot_ppl - teacher_ppl):+.2f}` |
| **Post-KD Restored Model** | **`{post_kd_ppl:.2f}`** | **`{post_kd_acc:.2f}%`** | **`{(post_kd_ppl - teacher_ppl):+.2f}`** |
{sst2_section}
### Hardware & Training Telemetry
- **Distillation Hardware:** NVIDIA RTX 3090 24GB (Sharif HPC Cluster)
- **Training Duration:** `{duration_str}`
- **Steps Completed:** `{steps}`
- **Throughput:** `{throughput:.1f} tokens/sec`
- **Peak GPU VRAM:** `{peak_vram} MB`

---

## 🚀 Quick Start & Usage

### 1. Using NeuralOps High-Level API (Recommended)

```python
import neuralops

# Instant load from Hugging Face Hub
model, tokenizer = neuralops.load_pretrained_compressed_bert(
    compressor_type="{compressor_name}",
    task="masked-lm",
)

text = "NeuralOps makes compressed models [MASK] and lightweight."
inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits
```

### 2. Standard Transformers Pipeline

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="{repo_id}")
results = unmasker("NeuralOps makes compressed transformers [MASK] and efficient.")
for res in results:
    print(f"{{res['token_str']}}: {{res['score']:.4f}}")
```

---

## 📜 Citation

```bibtex
@software{{neuralops2026compressorzoo,
  title   = {{NeuralOps: Pretrained BERT & Generative Model Compression Zoo}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}}
}}
```
"""


def list_pretrained_compressed_bert_models(
    metrics_path: Optional[str] = None,
    hf_org: str = "Hooshaai",
) -> list[dict[str, Any]]:
    """
    Returns the list of available pretrained compressed BERT models,
    including their parameter reduction ratio, PPL, accuracy, and Hugging Face repository IDs.
    """
    if metrics_path is None:
        metrics_path = "results/compressor_zoo/metrics.jsonl"

    models = []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        comp = record.get("compressor") or record.get("compressor_key")
                        if comp:
                            slug = comp.replace("_", "-")
                            record["repo_id"] = f"{hf_org}/bert-comp-{slug}-sst2"
                            models.append(record)
                    except Exception:
                        continue
    else:
        # Fallback to loading registered keys
        try:
            from neuralops.core.registry import COMPRESSOR_REGISTRY
            for k in sorted(COMPRESSOR_REGISTRY.keys()):
                slug = k.replace("_", "-")
                models.append({
                    "compressor": k,
                    "repo_id": f"{hf_org}/bert-comp-{slug}-sst2",
                    "status": "REGISTERED",
                })
        except Exception:
            pass

    return models


def load_pretrained_compressed_bert(
    compressor_type: str,
    task: Optional[str] = None,
    num_labels: int = 2,
    hf_org: str = "Hooshaai",
    device: str = "auto",
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """
    Downloads and instantiates a pretrained, compressed and KD-distilled BERT model
    directly from Hugging Face Hub.

    Supported tasks (auto-detected from config.json when ``task=None``):
      - "sequence-classification": AutoModelForSequenceClassification
      - "masked-lm": AutoModelForMaskedLM
      - "token-classification": AutoModelForTokenClassification
      - "feature-extraction": AutoModel
    """
    from neuralops.hub.surgery_loader import load_surgeried_model

    slug = compressor_type.replace("_", "-")
    repo_id = f"{hf_org}/bert-comp-{slug}-sst2"
    logger.info("Loading pretrained compressed %s from %s...", task or "auto", repo_id)
    return load_surgeried_model(
        repo_id=repo_id,
        device=device,
        task=task,
        num_labels=num_labels,
        **kwargs,
    )
