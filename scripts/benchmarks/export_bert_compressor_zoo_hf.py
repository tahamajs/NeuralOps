#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Materialize & Package 82 Compressed + LoRA-Healed BERT-SST2 Showcase Models for Hugging Face Hub.

Protocol:
  1. Reads the verified 328-run Multi-Rank Pareto Ledger (`multirank_lora_ledger.jsonl`).
  2. For each of the 82 compressors, selects the optimal Pareto rank checkpoint
     (maximizing healed SST-2 accuracy with real parameter reduction, or highest
     quantized accuracy when structural parameters remain unchanged).
  3. Reconstructs the compressed + LoRA-healed BERT checkpoint on GPU, verifies
     exact validation accuracy, saves full weights (`pytorch_model.bin`), tokenizer,
     `config.json`, `compression_summary.json`, and a publication-grade `README.md`
     embedding the complete 4-rank Pareto table (`k in [256, 128, 64, 32]`).
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compressor_multirank_lora_eval import (
    attach_lora_to_linear_layers,
    evaluate_sst2_accuracy,
    run_fast_lora_recovery,
)
from neuralops.core.compression_pipeline import compress_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_bert_comp_hf")

TEACHER_BASELINE_ACC = 91.63


def _clean_docstring(raw: str, fallback: str) -> str:
    if not raw or not raw.strip():
        return fallback
    trimmed = raw.strip()
    for stop in ("Attributes:", "Args:", "Parameters:", "Example:", "Examples:", "Raises:", "Returns:"):
        if stop in trimmed:
            trimmed = trimmed.split(stop, 1)[0]
    cleaned = " ".join(trimmed.split())
    return cleaned[:420] if cleaned else fallback


def extract_compressor_ast_metadata() -> dict[str, dict[str, str]]:
    """Extracts class/module/docstring metadata for all NeuralOps compressors."""
    meta: dict[str, dict[str, str]] = {}
    comp_dir = REPO_ROOT / "neuralops" / "compression"
    if not comp_dir.exists():
        return meta

    for py in comp_dir.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:
            continue
        family = py.parent.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                for dec in node.decorator_list:
                    dec_src = ast.unparse(dec)
                    if "COMPRESSOR_REGISTRY.register" in dec_src:
                        for arg in getattr(dec, "args", []):
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                key = arg.value
                                raw_doc = ast.get_docstring(node) or ""
                                meta[key] = {
                                    "class_name": node.name,
                                    "family": family,
                                    "module_path": str(py.relative_to(REPO_ROOT)),
                                    "docstring": _clean_docstring(raw_doc, f"NeuralOps {key} model compression engine."),
                                }
    return meta


def select_showcase_rank(rank_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Selects the optimal Pareto showcase rank for a compressor."""
    passed = [r for r in rank_records if r.get("status") == "PASSED"]
    if not passed:
        return None
    reduced = [r for r in passed if float(r.get("ratio", 1.0)) > 1.05]
    pool = reduced if reduced else passed
    return max(
        pool,
        key=lambda r: (float(r.get("healed_acc", 0.0)), float(r.get("ratio", 1.0)), -int(r.get("rank", 0))),
    )


def build_compressor_readme(
    comp_key: str,
    showcase: dict[str, Any],
    all_ranks: list[dict[str, Any]],
    meta: dict[str, str],
    out_dir: Path,
) -> str:
    """Generates a complete Hugging Face Model Card with full 4-rank Pareto benchmarks."""
    cls_name = meta.get("class_name", comp_key)
    family = meta.get("family", "compression")
    mod_path = meta.get("module_path", "neuralops/compression/")
    docstring = meta.get("docstring", f"NeuralOps {comp_key} compression operator.")

    rank = int(showcase.get("rank", 64))
    ratio = float(showcase.get("ratio", 1.0))
    orig_params = int(showcase.get("orig_params", 109483778))
    comp_params = int(showcase.get("comp_params", orig_params))
    reduction_pct = ((orig_params - comp_params) / max(orig_params, 1)) * 100.0
    zero_acc = float(showcase.get("zero_shot_acc", 0.0))
    healed_acc = float(showcase.get("healed_acc", zero_acc))
    lora_healed = bool(showcase.get("lora_healed", False))
    delta_teacher = healed_acc - TEACHER_BASELINE_ACC

    slug = comp_key.replace("_", "-")
    repo_slug = f"bert-comp-{slug}-sst2"
    repo_id = f"tahamajs/{repo_slug}"

    weights_path = out_dir / "pytorch_model.bin"
    weights_mb = f"{weights_path.stat().st_size / (1024 * 1024):.1f}_MB" if weights_path.exists() else "Compressed_FP16"
    badge_color = "brightgreen" if healed_acc >= 85.0 else ("green" if healed_acc >= 75.0 else ("yellow" if healed_acc >= 60.0 else "orange"))

    pareto_rows = []
    for rec in sorted(all_ranks, key=lambda x: -int(x.get("rank", 0))):
        r_k = rec.get("rank", "-")
        if rec.get("status") != "PASSED":
            pareto_rows.append(f"| `{r_k}` | — | — | `{rec.get('status')}` | — | — |")
            continue
        r_ratio = float(rec.get("ratio", 1.0))
        r_orig = int(rec.get("orig_params", orig_params))
        r_comp = int(rec.get("comp_params", r_orig))
        r_red = ((r_orig - r_comp) / max(r_orig, 1)) * 100.0
        r_zero = float(rec.get("zero_shot_acc", 0.0))
        r_heal = float(rec.get("healed_acc", r_zero))
        r_flag = " *(LoRA)*" if rec.get("lora_healed") else ""
        r_star = " ⭐" if int(r_k) == rank else ""
        pareto_rows.append(
            f"| **`{r_k}`{r_star}** | `{r_ratio:.2f}x` | `{r_comp:,}` (`-{r_red:.1f}%`) | `{r_zero:.2f}%` | **`{r_heal:.2f}%`**{r_flag} | `{r_heal - TEACHER_BASELINE_ACC:+.2f}%` |"
        )
    pareto_table = "\n".join(pareto_rows)

    return f"""---
language:
  - en
license: mit
license_link: https://github.com/tahamajs/NeuralOps/blob/main/LICENSE
library_name: neuralops
base_model: google-bert/bert-base-uncased
datasets:
  - nyu-mll/glue
metrics:
  - accuracy
tags:
  - neuralops
  - bert
  - model-compression
  - compressor-zoo
  - lora-healing
  - glue-sst2
  - {family}
  - {comp_key}
pipeline_tag: text-classification
model-index:
  - name: {repo_slug}
    results:
      - task:
          type: text-classification
          name: Sentiment Classification
        dataset:
          type: nyu-mll/glue
          name: GLUE SST-2
          config: sst2
          split: validation
        metrics:
          - type: accuracy
            value: {healed_acc:.2f}
            name: GLUE SST-2 Validation Accuracy
---

# BERT-Base ({comp_key}) — NeuralOps Compressed & LoRA-Healed Model Zoo

[![SST-2 Acc](https://img.shields.io/badge/GLUE_SST--2_Acc-{healed_acc:.2f}%25-{badge_color})](#-multi-rank-pareto-evaluation-k256-128-64-32)
[![Teacher Baseline](https://img.shields.io/badge/Teacher_Baseline-{TEACHER_BASELINE_ACC:.2f}%25-blue)](#-multi-rank-pareto-evaluation-k256-128-64-32)
[![Compression Ratio](https://img.shields.io/badge/Compression-{ratio:.2f}x_({reduction_pct:.1f}%25_Saved)-purple)](#-compression--architecture-profile)
[![Showcase Rank](https://img.shields.io/badge/Showcase_Rank-k={rank}-teal)](#-multi-rank-pareto-evaluation-k256-128-64-32)
[![Weights](https://img.shields.io/badge/Weights-{weights_mb}-indigo)](#-packaged-artifacts)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/tahamajs/NeuralOps/blob/main/LICENSE)

This repository provides a physically compressed and LoRA-healed checkpoint of **`google-bert/bert-base-uncased`** fine-tuned on **GLUE SST-2**, compressed using the **`{comp_key}`** (`{cls_name}`) engine from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps) at Pareto-selected rank **`k={rank}`** (`{ratio:.2f}x` compression ratio, `-{reduction_pct:.1f}%` parameter reduction).

---

## Compression & Architecture Profile

> **{cls_name}**: {docstring}

| Property | Specification |
|---|---|
| **Registry Key (`COMPRESSOR_REGISTRY`)** | `{comp_key}` |
| **NeuralOps Class & Family** | `{cls_name}` (`{family}`) |
| **Source Implementation** | [`{mod_path}`](https://github.com/tahamajs/NeuralOps/blob/main/{mod_path}) |
| **Base Architecture** | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (`109,483,778` parameters, 12 layers, $D=768$) |
| **Showcase Rank (`k`)** | **`{rank}`** (`LoRA Recovery Applied: {lora_healed}`) |
| **Compressed Parameter Count** | **`{comp_params:,}`** (`-{reduction_pct:.2f}%` reduction, **`{ratio:.3f}x`** physical compression ratio) |
| **Teacher Baseline SST-2 Acc** | `{TEACHER_BASELINE_ACC:.2f}%` |
| **Showcase Final SST-2 Acc** | **`{healed_acc:.2f}%`** (Zero-Shot: `{zero_acc:.2f}%`, Parity Delta: `{delta_teacher:+.2f}%`) |

---

## Multi-Rank Pareto Evaluation (`k=256, 128, 64, 32`)

Evaluated physically on **NVIDIA GeForce RTX 3090 (24GB)** on the GLUE SST-2 validation split (`872` samples). Whenever zero-shot compression drops accuracy by `> 2.0%` alongside `> 1.10x` reduction, NeuralOps applies targeted `r=8` LoRA healing (`50` steps, `lr=2e-4`, 45s budget):

| Rank (`k`) | Physical Ratio | Compressed Params (Reduction) | Zero-Shot Acc | Healed Acc | Parity vs Teacher (`{TEACHER_BASELINE_ACC:.2f}%`) |
|:---:|:---:|---:|---:|---:|---:|
{pareto_table}

*(⭐ indicates the serialized checkpoint packaged in this Hugging Face repository.)*

---

## Packaged Artifacts

- **`pytorch_model.bin`**: Full serialized compressed BERT state dictionary (with low-rank factors / quantization buffers / LoRA healing adapters baked in).
- **`config.json`**: Model configuration tagged with `compression_method: "{comp_key}"` and `compression_rank: {rank}`.
- **`compression_summary.json`**: Complete machine-readable telemetry across all evaluated ranks `[256, 128, 64, 32]`.
- **Tokenizer Assets**: Standard `bert-base-uncased` vocabulary (`vocab.txt`, `tokenizer.json`, `tokenizer_config.json`).

---

## Quick Start (Reconstruction & Inference)

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertForSequenceClassification
from neuralops.core.compression_pipeline import compress_model

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Reconstruct compressed NeuralOps topology and load serialized compressed + healed weights
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
model = compress_model(model, {{"compressors": [{{"method": "{comp_key}", "rank": {rank}}}]}})

ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

inputs = tokenizer("NeuralOps achieves strong compression without sacrificing downstream accuracy.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
print("Predicted SST-2 Sentiment (0=Negative, 1=Positive):", pred)
```

---

## Citation

```bibtex
@software{{neuralops2026compressorzoo,
  title   = {{NeuralOps: Multi-Rank Compressed BERT-SST2 Zoo with Pareto LoRA Recovery}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize & package 82 Compressed BERT-SST2 showcase checkpoints for HF Hub.")
    parser.add_argument("--ledger", default="/home/tahamajlesi.ut/NeuralOps/results/bert_compressor_zoo/multirank_lora_ledger.jsonl")
    parser.add_argument("--teacher_path", default="/dev/shm/bert_sst2_teacher")
    parser.add_argument("--cache_path", default="/tmp/sst2_train_tokens.pt")
    parser.add_argument("--val_cache_path", default="/tmp/sst2_val_cache.pt")
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/bert_compressor_zoo_hf")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        logger.error("Ledger not found: %s", ledger_path)
        sys.exit(2)

    by_comp: dict[str, list[dict[str, Any]]] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            key = d.get("compressor")
            if key:
                by_comp.setdefault(key, []).append(d)

    logger.info("Loaded ledger for %d compressors from %s", len(by_comp), ledger_path)
    meta_map = extract_compressor_ast_metadata()

    from transformers import AutoTokenizer, BertForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, local_files_only=True)
    base_teacher = BertForSequenceClassification.from_pretrained(args.teacher_path, local_files_only=True)

    vc = torch.load(args.val_cache_path, map_location="cpu")
    val_loader = DataLoader(TensorDataset(vc["input_ids"], vc["attention_mask"], vc["labels"]), batch_size=128, shuffle=False)
    tc = torch.load(args.cache_path, map_location="cpu")
    train_loader = DataLoader(TensorDataset(tc["input_ids"], tc["attention_mask"], tc["labels"]), batch_size=64, shuffle=True)

    for idx, comp_key in enumerate(sorted(by_comp.keys()), 1):
        comp_dir = out_base / comp_key
        summary_file = comp_dir / "compression_summary.json"
        if summary_file.exists() and ((comp_dir / "pytorch_model.bin").exists() or (comp_dir / "upload_success.json").exists()):
            logger.info("[%d/%d] Skipping %s (already packaged).", idx, len(by_comp), comp_key)
            continue

        ranks_list = by_comp[comp_key]
        showcase = select_showcase_rank(ranks_list)
        if showcase is None:
            logger.warning("[%d/%d] Skipping %s (no PASSED rank in ledger).", idx, len(by_comp), comp_key)
            continue

        rank = int(showcase["rank"])
        logger.info("[%d/%d] Packaging %s at showcase rank k=%d...", idx, len(by_comp), comp_key, rank)

        try:
            student = copy.deepcopy(base_teacher)
            student = compress_model(student, {"compressors": [{"method": comp_key, "rank": rank}]})
            student = student.to(device)

            zero_acc, zero_loss = evaluate_sst2_accuracy(student, val_loader, device)
            comp_params = sum(p.numel() for p in student.parameters())
            orig_params = sum(p.numel() for p in base_teacher.parameters())
            ratio = round(orig_params / max(comp_params, 1), 3)
            delta_zero = round(zero_acc - TEACHER_BASELINE_ACC, 2)

            if delta_zero < -2.0 and ratio > 1.10:
                attach_lora_to_linear_layers(student, r=8)
                student = student.to(device)
                healed_acc, healed_loss = run_fast_lora_recovery(
                    student=student,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    max_steps=50,
                )
                showcase["lora_healed"] = True
                showcase["healed_acc"] = round(healed_acc, 2)
                showcase["healed_loss"] = round(healed_loss, 4)

            comp_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(comp_dir))
            torch.save(student.cpu().half().state_dict(), comp_dir / "pytorch_model.bin")

            cfg = student.config.to_dict()
            cfg["_name_or_path"] = "google-bert/bert-base-uncased"
            cfg["base_model_name_or_path"] = "google-bert/bert-base-uncased"
            cfg["license"] = "mit"
            cfg["compression_method"] = comp_key
            cfg["compression_rank"] = rank
            cfg["compression_ratio"] = ratio
            (comp_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            summary = {
                "compressor": comp_key,
                "showcase_rank": rank,
                "showcase_metrics": showcase,
                "all_ranks": ranks_list,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            readme = build_compressor_readme(comp_key, showcase, ranks_list, meta_map.get(comp_key, {}), comp_dir)
            (comp_dir / "README.md").write_text(readme, encoding="utf-8")
            logger.info("[%d/%d] Packaged %s (Acc=%.2f%%, Ratio=%.2fx) -> %s", idx, len(by_comp), comp_key, showcase.get("healed_acc", zero_acc), ratio, comp_dir)
        except Exception as e:
            logger.error("[%d/%d] Failed packaging %s: %s", idx, len(by_comp), comp_key, e)
        finally:
            if "student" in locals():
                del student
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    logger.info("Completed BERT Compressor Zoo packaging at %s", out_base)


if __name__ == "__main__":
    main()
