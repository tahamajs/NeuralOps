import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

sys.path.append("Codes")
from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NeuralOpsServerPipeline")

# 15 Modules from Technical Specification Document
SPEC_MODULES = [
    # Phase I: Calibration & Pre-Whitening
    "asvd_attention",
    "svd_llm_v2_attention",
    # Phase II: Adaptive Rank Allocation
    "adaptive_svd_attention",
    "squeezing_heads",
    "gqa_attention",
    # Phase III: Core Attention & SSM Engines
    "primal_attention",
    "svd_linear_attention",
    "mamba_svd_flash",
    "res_svd",
    # Phase IV: Recovery Fine-Tuning
    "lora_attention",
    "squeezing_heads_distill",
    "fusion_attn",
    # Phase V: Hardware Serving
    "fused_svd_kernel",
    "flash_attention3",
    "spectron",
]

def load_eval_data(dataset_name: str, tokenizer, max_samples: int = 500, seq_len: int = 128):
    ds = load_dataset("glue", dataset_name, split=f"validation[:{max_samples}]")
    if dataset_name == "sst2":
        texts = list(ds["sentence"])
        labels = list(ds["label"])
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    elif dataset_name == "mrpc":
        s1 = list(ds["sentence1"])
        s2 = list(ds["sentence2"])
        labels = list(ds["label"])
        enc = tokenizer(s1, s2, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return enc, torch.tensor(labels, dtype=torch.long)

def load_train_loader(dataset_name: str, tokenizer, batch_size: int = 16, max_samples: int = 1000, seq_len: int = 128):
    ds = load_dataset("glue", dataset_name, split=f"train[:{max_samples}]")
    if dataset_name == "sst2":
        texts = list(ds["sentence"])
        labels = list(ds["label"])
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    elif dataset_name == "mrpc":
        s1 = list(ds["sentence1"])
        s2 = list(ds["sentence2"])
        labels = list(ds["label"])
        enc = tokenizer(s1, s2, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    
    tensor_ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=True)

def evaluate_accuracy(model, eval_enc, eval_labels, batch_size: int = 32, device="cuda"):
    model.eval()
    input_ids = eval_enc["input_ids"]
    attention_mask = eval_enc["attention_mask"]
    num_samples = input_ids.size(0)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            b_ids = input_ids[i:i+batch_size].to(device)
            b_mask = attention_mask[i:i+batch_size].to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask)
            preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
    acc = accuracy_score(eval_labels.tolist(), all_preds) * 100.0
    f1 = f1_score(eval_labels.tolist(), all_preds, zero_division=0)
    return round(acc, 2), round(float(f1), 4)

def run_recovery_tuning(model, train_loader, recovery_steps: int = 50, lr: float = 1e-4, device="cuda"):
    # Freeze base weights, train adapters and/or classification head
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("lora", "classifier", "pre_classifier", "adapter", "dense", "head")):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    if not trainable_params:
        # Fallback: train all parameters if no specific adapter found
        trainable_params = list(model.parameters())
        for p in trainable_params:
            p.requires_grad = True

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    model.train()
    
    step = 0
    while step < recovery_steps:
        for b_ids, b_mask, b_labels in train_loader:
            if step >= recovery_steps:
                break
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = out.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "mrpc"])
    parser.add_argument("--model_path", type=str, default="D:/SVD_Project/models/distilbert_sst2")
    parser.add_argument("--output_dir", type=str, default="D:/SVD_Project/results/neuralops_suite")
    parser.add_argument("--recovery_steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info(f"🌟 NEURALOPS PIPELINE: {args.dataset.upper()} ON GPU")
    logger.info(f"Base Checkpoint: {args.model_path}")
    logger.info(f"Recovery Steps: {args.recovery_steps} (lr=1e-4) | Target Rank: {args.rank}")
    logger.info("=================================================================")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eval_enc, eval_labels = load_eval_data(args.dataset, tokenizer)
    train_loader = load_train_loader(args.dataset, tokenizer)

    # 1. Base Model Verification
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    base_acc, base_f1 = evaluate_accuracy(base_model, eval_enc, eval_labels, device=device)
    logger.info(f"Verified Base Model Accuracy on {args.dataset}: {base_acc}% | F1: {base_f1}")
    del base_model
    torch.cuda.empty_cache()

    results = []

    for mod_name in SPEC_MODULES:
        logger.info(f"\n--- Testing Module: {mod_name} ---")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
        
        # Inject attention module with weight copying
        try:
            inject_attention(model, mod_name, rank=args.rank, copy_weights=True)
            logger.info(f"Successfully injected {mod_name} with copy_weights=True")
        except Exception as e:
            logger.error(f"Failed to inject {mod_name}: {e}")
            results.append({
                "module": mod_name,
                "dataset": args.dataset,
                "status": "FAIL",
                "error": str(e),
                "zero_shot_acc": 0.0,
                "post_recovery_acc": 0.0,
                "quality_gate": "FAIL",
            })
            del model
            torch.cuda.empty_cache()
            continue

        try:
            # Measure Zero-Shot Accuracy
            zero_shot_acc, zero_shot_f1 = evaluate_accuracy(model, eval_enc, eval_labels, device=device)
            logger.info(f"[{mod_name}] Zero-Shot Accuracy: {zero_shot_acc}% | F1: {zero_shot_f1}")

            # Run Recovery Fine-Tuning (50 steps)
            t0 = time.perf_counter()
            run_recovery_tuning(model, train_loader, recovery_steps=args.recovery_steps, lr=1e-4, device=device)
            tune_time = time.perf_counter() - t0

            # Measure Post-Recovery Accuracy
            post_acc, post_f1 = evaluate_accuracy(model, eval_enc, eval_labels, device=device)
            logger.info(f"[{mod_name}] Post-Recovery Accuracy: {post_acc}% | F1: {post_f1} (Tuned in {tune_time:.1f}s)")

            # Quality Gate Check (Hard Rule: > 80% on SST-2)
            gate = "PASS" if (args.dataset == "sst2" and post_acc >= 80.0) or (args.dataset != "sst2" and post_acc >= 70.0) else "FAIL (<80%)"

            record = {
                "module": mod_name,
                "dataset": args.dataset,
                "status": "PASS",
                "zero_shot_acc": zero_shot_acc,
                "zero_shot_f1": zero_shot_f1,
                "post_recovery_acc": post_acc,
                "post_recovery_f1": post_f1,
                "delta_acc": round(post_acc - zero_shot_acc, 2),
                "quality_gate": gate,
            }
            results.append(record)

        except Exception as e:
            logger.error(f"Execution failed for {mod_name}: {e}")
            results.append({
                "module": mod_name,
                "dataset": args.dataset,
                "status": "FAIL",
                "error": str(e),
                "zero_shot_acc": 0.0,
                "post_recovery_acc": 0.0,
                "quality_gate": "FAIL",
            })
        finally:
            # Save incremental CSV
            df = pd.DataFrame(results)
            df.to_csv(out_dir / f"neuralops_{args.dataset}_results.csv", index=False)
            del model
            torch.cuda.empty_cache()

    logger.info(f"\nCompleted all {len(SPEC_MODULES)} modules on {args.dataset}! Results saved to: {out_dir / f'neuralops_{args.dataset}_results.csv'}")

if __name__ == "__main__":
    main()
