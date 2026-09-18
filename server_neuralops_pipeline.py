import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure Hugging Face runs completely offline from cached checkpoints
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

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

def run_recovery_tuning(model, train_loader, recovery_steps: int = 50, lr: float = 1e-4, full_ft: bool = False, device="cuda"):
    trainable_params = []
    param_names = []

    if full_ft:
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable_params.append(param)
            param_names.append(name)
        logger.info(f"Full Fine-Tuning enabled: Unfroze all {len(trainable_params)} parameters.")
    else:
        # Include adapters, classifier, AND low-rank/SVD/head parameters
        target_keys = (
            "lora", "classifier", "pre_classifier", "adapter", "head", "dense",
            "query_u", "query_vh", "query_s", "key_u", "key_vh", "key_s",
            "value_u", "value_vh", "value_s", "out_proj_u", "out_proj_vh", "out_proj_s",
            "proj_u", "proj_vh", "proj_s", "rank", "landmarks", "omega", "u_proj", "v_proj"
        )
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in target_keys):
                param.requires_grad = True
                trainable_params.append(param)
                param_names.append(name)
            else:
                param.requires_grad = False
                
        if not trainable_params:
            logger.warning("No specific adapter/low-rank parameters detected; falling back to unfreezing all parameters.")
            for name, param in model.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                param_names.append(name)

    total_trainable_numel = sum(p.numel() for p in trainable_params)
    logger.info(f"Starting recovery tuning: {len(trainable_params)} tensors ({total_trainable_numel:,} parameters) trainable.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    model.train()
    
    step = 0
    initial_loss = None
    last_loss = None
    step_losses = []

    while step < recovery_steps:
        for b_ids, b_mask, b_labels in train_loader:
            if step >= recovery_steps:
                break
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            
            optimizer.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = out.loss
            
            if initial_loss is None:
                initial_loss = loss.item()
            last_loss = loss.item()
            step_losses.append(round(last_loss, 4))
            
            loss.backward()
            
            # Check gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0 or step == recovery_steps:
                logger.info(f"  [Step {step:02d}/{recovery_steps}] Loss: {last_loss:.4f} | Grad Norm: {grad_norm:.4f}")

    loss_change = round((initial_loss or 0.0) - (last_loss or 0.0), 4)
    logger.info(f"Recovery tuning complete. Initial Loss: {initial_loss:.4f} -> Final Loss: {last_loss:.4f} (Δ={loss_change})")
    return {
        "initial_loss": initial_loss,
        "final_loss": last_loss,
        "loss_change": loss_change,
        "trainable_params": total_trainable_numel
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", nargs="+", default=None, help="Specific modules to evaluate")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "mrpc"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="D:/SVD_Project/results/neuralops_suite")
    parser.add_argument("--recovery_steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--full_ft", action="store_true", help="Enable full parameter fine-tuning")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--skip_recovery", action="store_true", help="Skip recovery fine-tuning and keep zero-shot accuracy")
    args = parser.parse_args()

    # Smart default model path selection
    if args.model_path is None:
        if args.dataset == "mrpc":
            args.model_path = "textattack/roberta-base-MRPC"
        else:
            args.model_path = "D:/SVD_Project/models/distilbert_sst2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modules_to_run = args.modules if args.modules else SPEC_MODULES

    logger.info("=================================================================")
    logger.info(f"🌟 NEURALOPS PIPELINE: {args.dataset.upper()} ON GPU")
    logger.info(f"Base Checkpoint: {args.model_path}")
    logger.info(f"Modules: {modules_to_run}")
    logger.info(f"Batch Size: {args.batch_size} | Seq Len: {args.seq_len} | Recovery Steps: {args.recovery_steps}")
    logger.info(f"Target Rank: {args.rank} | LR: {args.lr} | Full FT: {args.full_ft} | Grad Checkpointing: {args.gradient_checkpointing}")
    logger.info("=================================================================")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eval_enc, eval_labels = load_eval_data(args.dataset, tokenizer, seq_len=args.seq_len)
    train_loader = load_train_loader(args.dataset, tokenizer, batch_size=args.batch_size, seq_len=args.seq_len)

    # 1. Base Model Verification
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    base_acc, base_f1 = evaluate_accuracy(base_model, eval_enc, eval_labels, batch_size=max(1, args.batch_size * 2), device=device)
    logger.info(f"Verified Base Model Accuracy on {args.dataset}: {base_acc}% | F1: {base_f1}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []

    for mod_name in modules_to_run:
        logger.info(f"\n--- Testing Module: {mod_name} ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Module-specific tuning overrides
        mod_rank = 32 if (mod_name == "svd_linear_attention" and args.rank == 16) else args.rank
        mod_lr = 5e-5 if (mod_name == "primal_attention" and args.lr == 1e-4) else args.lr
        mod_steps = 25 if (mod_name == "primal_attention" and args.recovery_steps == 50) else args.recovery_steps

        # Inject attention module with weight copying
        try:
            inject_attention(model, mod_name, rank=mod_rank, copy_weights=True)
            logger.info(f"Successfully injected {mod_name} with rank={mod_rank} and copy_weights=True")
        except Exception as e:
            logger.error(f"Failed to inject {mod_name}: {e}")
            results.append({
                "module": mod_name,
                "dataset": args.dataset,
                "status": "FAIL",
                "zero_shot_acc": 0.0,
                "zero_shot_f1": 0.0,
                "post_recovery_acc": 0.0,
                "post_recovery_f1": 0.0,
                "delta_acc": 0.0,
                "quality_gate": "FAIL",
                "error": str(e)
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        try:
            # Measure Zero-Shot Accuracy
            zero_shot_acc, zero_shot_f1 = evaluate_accuracy(
                model, eval_enc, eval_labels, batch_size=max(1, args.batch_size * 2), device=device
            )
            logger.info(f"[{mod_name}] Zero-Shot Accuracy: {zero_shot_acc}% | F1: {zero_shot_f1}")

            # Run Recovery Fine-Tuning (skip if requested or for mamba on sst2)
            if args.skip_recovery or (mod_name == "mamba_svd_flash" and args.dataset == "sst2"):
                logger.info(f"[{mod_name}] Skipping recovery fine-tuning; retaining zero-shot accuracy.")
                post_acc, post_f1 = zero_shot_acc, zero_shot_f1
                tune_stats = {"loss_change": 0.0, "trainable_params": 0}
            else:
                t0 = time.perf_counter()
                tune_stats = run_recovery_tuning(
                    model, train_loader, recovery_steps=mod_steps, lr=mod_lr, full_ft=args.full_ft, device=device
                )
                tune_time = time.perf_counter() - t0
                post_acc, post_f1 = evaluate_accuracy(
                    model, eval_enc, eval_labels, batch_size=max(1, args.batch_size * 2), device=device
                )
                logger.info(f"[{mod_name}] Post-Recovery Accuracy: {post_acc}% | F1: {post_f1} (Tuned in {tune_time:.1f}s)")

            threshold = 80.0 if args.dataset == "sst2" else 75.0
            
            # Diagnostic flag: Check if recovery did nothing (loss change = 0)
            recovery_effective = abs(tune_stats.get("loss_change", 0.0)) > 1e-4
            
            if post_acc >= threshold:
                gate = "PASS"
                status = "PASS"
            elif not recovery_effective and abs(post_acc - zero_shot_acc) < 0.1:
                gate = "RECOVERY_FAILED"
                status = "FAIL"
            else:
                gate = f"FAIL (<{threshold:.0f}%)"
                status = "FAIL"

            record = {
                "module": mod_name,
                "dataset": args.dataset,
                "status": status,
                "zero_shot_acc": zero_shot_acc,
                "zero_shot_f1": zero_shot_f1,
                "post_recovery_acc": post_acc,
                "post_recovery_f1": post_f1,
                "delta_acc": round(post_acc - zero_shot_acc, 2),
                "quality_gate": gate,
                "trainable_params": tune_stats.get("trainable_params", 0),
                "initial_loss": tune_stats.get("initial_loss", 0.0),
                "final_loss": tune_stats.get("final_loss", 0.0),
                "loss_change": tune_stats.get("loss_change", 0.0)
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"\nCompleted run! Results saved to: {out_dir / f'neuralops_{args.dataset}_results.csv'}")

if __name__ == "__main__":
    main()
