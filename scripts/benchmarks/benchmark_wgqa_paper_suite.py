#!/usr/bin/env python3
"""
Empirical Benchmark Suite for WGQA Paper:
"Weighted Grouped-Query Attention: A Cross-Architecture Study"

Evaluates:
  - Backbones: DistilBERT, RoBERTa, GPT-2
  - Tasks: SST-2, MNLI, AG News
  - Attention Variants: Softmax (MHA baseline), GQA, WGQA
  - 3 Seeds: 42, 123, 456
  - LoRA Rank Ablation on WGQA: r in {8, 16, 32, 64}

Runs strictly offline on NVIDIA RTX 4070 (SM 8.9).
Outputs:
  - results/wgqa_paper/table1_cross_architecture.json
  - results/wgqa_paper/table2_multiseed_robustness.json
  - results/wgqa_paper/table3_lora_ablation.json
  - results/wgqa_paper/wgqa_all_raw_metrics.json
"""

import os
import time
import json
import logging
from pathlib import Path

# Enforce strict offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset

from neuralops.attentions.modern.gqa_attention import inject_gqa
from neuralops.attentions.modern.wgqa_attention import inject_wgqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WGQAPaperSuite")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results/wgqa_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "distilbert": {
        "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
        "mnli": "distilbert-base-uncased",
        "ag_news": "distilbert-base-uncased",
        "base": "distilbert-base-uncased",
    },
    "roberta": {
        "sst2": "textattack/roberta-base-SST-2",
        "mnli": "roberta-base",
        "ag_news": "roberta-base",
        "base": "roberta-base",
    },
    "gpt2": {
        "sst2": "gpt2",
        "mnli": "gpt2",
        "ag_news": "gpt2",
        "base": "gpt2",
    },
}


def set_seed(seed: int):
    """Execute set seed routine."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prepare_batches(task: str, tokenizer, max_samples: int = 1000, batch_size: int = 32):
    """Execute prepare batches routine."""
    if task == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        texts = [x["sentence"] for x in ds]
        labels = [x["label"] for x in ds]
    elif task == "mnli":
        ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        sep = tokenizer.sep_token if tokenizer.sep_token else " "
        texts = [f"{x['premise']} {sep} {x['hypothesis']}" for x in ds]
        labels = [x["label"] for x in ds]
    elif task == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split="test")
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        texts = [x["text"] for x in ds]
        labels = [x["label"] for x in ds]
    else:
        raise ValueError(f"Unknown task: {task}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    batches = []
    for i in range(0, len(texts), batch_size):
        b_texts = texts[i : i + batch_size]
        b_labels = labels[i : i + batch_size]
        enc = tokenizer(
            b_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        y = torch.tensor(b_labels, device=DEVICE, dtype=torch.long)
        batches.append((enc, y))
    return batches


def evaluate_classification(model, batches):
    """Execute evaluate classification routine."""
    model.eval()
    correct = 0
    total = 0
    total_tokens = 0
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.no_grad():
        for enc, labels in batches:
            outputs = model(**enc)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_tokens += enc["input_ids"].numel()

    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    acc = 100.0 * correct / max(total, 1)
    tok_per_sec = total_tokens / max(elapsed, 1e-6)
    ms_per_token = (elapsed * 1000.0) / max(total_tokens, 1)

    return {
        "accuracy": round(acc, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "throughput_tok_per_sec": round(tok_per_sec, 2),
        "latency_ms_per_token": round(ms_per_token, 4),
        "elapsed_s": round(elapsed, 3),
        "num_samples": total,
    }


def count_params(model):
    """Execute count params routine."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(backbone: str, task: str, variant: str, seed: int, lora_rank: int = 16):
    """Execute build model routine."""
    set_seed(seed)
    model_name = MODEL_CONFIGS[backbone].get(task, MODEL_CONFIGS[backbone]["base"])
    num_labels = 2 if task == "sst2" else (3 if task == "mnli" else 4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    total_before, _ = count_params(model)

    is_dec = backbone == "gpt2"
    if variant == "softmax":
        pass
    elif variant == "gqa":
        inject_gqa(model=model, num_kv_heads=4, is_decoder=is_dec)
    elif variant == "wgqa":
        inject_wgqa(model=model, num_kv_heads=4, lora_rank=lora_rank, is_decoder=is_dec)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model.to(DEVICE)
    total_after, trainable = count_params(model)
    return model, tokenizer, total_before, total_after, trainable


def run_suite():
    """Execute run suite routine."""
    backbones = ["distilbert", "roberta", "gpt2"]
    tasks = ["sst2", "mnli", "ag_news"]
    variants = ["softmax", "gqa", "wgqa"]
    seeds = [42, 123, 456]
    lora_ranks = [8, 16, 32, 64]

    raw_results = []

    logger.info("STARTING WGQA PAPER EMPIRICAL BENCHMARK SUITE")
    logger.info("Hardware: NVIDIA RTX 4070 (SM 8.9)")

    for b_idx, backbone in enumerate(backbones, 1):
        for t_idx, task in enumerate(tasks, 1):
            logger.info(f"--> [{b_idx}/{len(backbones)}] Backbone: {backbone} | Task: {task}")
            for variant in variants:
                for seed in seeds:
                    logger.info(f"Running {backbone} | {task} | {variant} | seed={seed}...")
                    model, tokenizer, p_before, p_after, p_train = build_model(
                        backbone, task, variant, seed, lora_rank=16
                    )
                    batches = prepare_batches(task, tokenizer, max_samples=872 if task == "sst2" else 1000)
                    metrics = evaluate_classification(model, batches)
                    del model
                    torch.cuda.empty_cache()

                    entry = {
                        "backbone": backbone,
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "lora_rank": 16,
                        "total_params_before": p_before,
                        "total_params_after": p_after,
                        "trainable_params": p_train,
                        **metrics,
                    }
                    raw_results.append(entry)
                    logger.info(
                        f"[RESULT] {backbone} | {task} | {variant} (s={seed}) => "
                        f"Acc: {metrics['accuracy']}%, VRAM: {metrics['peak_vram_mb']}MB, "
                        f"Throughput: {metrics['throughput_tok_per_sec']} tok/s"
                    )

    # 2. LoRA Rank Ablation on DistilBERT SST-2
    logger.info("--> Running LoRA Rank Ablation on DistilBERT SST-2...")
    lora_ablation_results = []
    for r in lora_ranks:
        logger.info(f"Evaluating WGQA LoRA rank r={r}...")
        model, tokenizer, p_before, p_after, p_train = build_model("distilbert", "sst2", "wgqa", seed=42, lora_rank=r)
        batches = prepare_batches("sst2", tokenizer, max_samples=872)
        metrics = evaluate_classification(model, batches)
        del model
        torch.cuda.empty_cache()

        abl_entry = {
            "rank": r,
            "total_params": p_after,
            "trainable_params": p_train,
            "accuracy": metrics["accuracy"],
            "peak_vram_mb": metrics["peak_vram_mb"],
            "throughput_tok_per_sec": metrics["throughput_tok_per_sec"],
            "latency_ms_per_token": metrics["latency_ms_per_token"],
        }
        lora_ablation_results.append(abl_entry)
        logger.info(f"[ABLATION] r={r}: Acc={metrics['accuracy']}%, VRAM={metrics['peak_vram_mb']}MB")

    # 3. Aggregate Table 1: Cross-Architecture
    table1 = []
    for backbone in backbones:
        for task in tasks:
            row = {"backbone": backbone, "task": task}
            for variant in variants:
                matches = [
                    r
                    for r in raw_results
                    if r["backbone"] == backbone and r["task"] == task and r["variant"] == variant
                ]
                accs = [m["accuracy"] for m in matches]
                vrams = [m["peak_vram_mb"] for m in matches]
                thrs = [m["throughput_tok_per_sec"] for m in matches]
                row[f"{variant}_acc_mean"] = round(float(np.mean(accs)), 2)
                row[f"{variant}_acc_std"] = round(float(np.std(accs)), 2)
                row[f"{variant}_vram_mb"] = round(float(np.mean(vrams)), 1)
                row[f"{variant}_throughput"] = round(float(np.mean(thrs)), 1)
            table1.append(row)

    # 4. Aggregate Table 2: Multi-Seed Robustness
    table2 = []
    for backbone in backbones:
        for task in tasks:
            wgqa_matches = [
                r for r in raw_results if r["backbone"] == backbone and r["task"] == task and r["variant"] == "wgqa"
            ]
            s42 = next(m["accuracy"] for m in wgqa_matches if m["seed"] == 42)
            s123 = next(m["accuracy"] for m in wgqa_matches if m["seed"] == 123)
            s456 = next(m["accuracy"] for m in wgqa_matches if m["seed"] == 456)
            all_acc = [s42, s123, s456]
            table2.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "seed_42": s42,
                    "seed_123": s123,
                    "seed_456": s456,
                    "mean": round(float(np.mean(all_acc)), 2),
                    "std": round(float(np.std(all_acc)), 2),
                }
            )

    with open(OUTPUT_DIR / "table1_cross_architecture.json", "w") as f:
        json.dump(table1, f, indent=2)
    with open(OUTPUT_DIR / "table2_multiseed_robustness.json", "w") as f:
        json.dump(table2, f, indent=2)
    with open(OUTPUT_DIR / "table3_lora_ablation.json", "w") as f:
        json.dump(lora_ablation_results, f, indent=2)
    with open(OUTPUT_DIR / "wgqa_all_raw_metrics.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    logger.info("BENCHMARK COMPLETE! Artifacts saved to results/wgqa_paper")


if __name__ == "__main__":
    run_suite()
