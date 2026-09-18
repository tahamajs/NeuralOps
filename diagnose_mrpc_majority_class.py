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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

sys.path.append("Codes")
from framework.core.attention_utils import inject_attention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MRPC_Diagnosis")

def load_data(tokenizer, max_eval: int = 500, max_train: int = 1000, seq_len: int = 128):
    eval_ds = load_dataset("glue", "mrpc", split=f"validation[:{max_eval}]")
    s1_eval = list(eval_ds["sentence1"])
    s2_eval = list(eval_ds["sentence2"])
    labels_eval = list(eval_ds["label"])
    eval_enc = tokenizer(s1_eval, s2_eval, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")

    train_ds = load_dataset("glue", "mrpc", split=f"train[:{max_train}]")
    s1_train = list(train_ds["sentence1"])
    s2_train = list(train_ds["sentence2"])
    labels_train = list(train_ds["label"])
    train_enc = tokenizer(s1_train, s2_train, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    
    tensor_ds = torch.utils.data.TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(labels_train, dtype=torch.long))
    train_loader = DataLoader(tensor_ds, batch_size=16, shuffle=True)
    
    return eval_enc, torch.tensor(labels_eval, dtype=torch.long), train_loader

def evaluate(model, eval_enc, eval_labels, device="cuda"):
    model.eval()
    input_ids = eval_enc["input_ids"]
    attention_mask = eval_enc["attention_mask"]
    num_samples = input_ids.size(0)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, num_samples, 32):
            b_ids = input_ids[i:i+32].to(device)
            b_mask = attention_mask[i:i+32].to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask)
            preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
    acc = accuracy_score(eval_labels.tolist(), all_preds) * 100.0
    f1 = f1_score(eval_labels.tolist(), all_preds, zero_division=0)
    cm = confusion_matrix(eval_labels.tolist(), all_preds).tolist()
    pred_counts = {str(k): int(all_preds.count(k)) for k in set(all_preds)}
    return round(acc, 2), round(float(f1), 4), cm, pred_counts

def main():
    model_path = "textattack/roberta-base-MRPC"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eval_enc, eval_labels, train_loader = load_data(tokenizer)

    out_base = Path("results/mrpc_test")
    out_base.mkdir(parents=True, exist_ok=True)

    diagnosis_results = []

    # 1. Base Model Check
    base = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    b_acc, b_f1, b_cm, b_preds = evaluate(base, eval_enc, eval_labels, device=device)
    logger.info(f"Verified Base Model MRPC: Acc={b_acc}%, F1={b_f1}, Preds={b_preds}, CM={b_cm}")
    del base
    torch.cuda.empty_cache()

    # 2. Test Ranks 64, 128, 256
    for rank in [64, 128, 256]:
        logger.info(f"\n==================================================")
        logger.info(f"🧪 DIAGNOSIS: svd_llm_v2_attention @ RANK={rank}")
        logger.info(f"==================================================")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        inject_attention(model, "svd_llm_v2_attention", rank=rank, copy_weights=True)
        
        zero_acc, zero_f1, zero_cm, zero_preds = evaluate(model, eval_enc, eval_labels, device=device)
        logger.info(f"[Rank {rank}] Zero-Shot: Acc={zero_acc}% | F1={zero_f1} | Preds={zero_preds}")

        # Recovery tuning (50 steps, lr=1e-4)
        target_keys = (
            "lora", "classifier", "pre_classifier", "adapter", "head",
            "query_u", "query_vh", "query_s", "key_u", "key_vh", "key_s",
            "value_u", "value_vh", "value_s", "out_proj_u", "out_proj_vh", "out_proj_s"
        )
        trainable_params = []
        trainable_names = []
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in target_keys):
                param.requires_grad = True
                trainable_params.append(param)
                trainable_names.append(name)
            else:
                param.requires_grad = False

        logger.info(f"[Rank {rank}] Trainable params: {len(trainable_params)} tensors ({sum(p.numel() for p in trainable_params):,} params)")
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        model.train()
        step = 0
        step_logs = []
        while step < 50:
            for b_ids, b_mask, b_labels in train_loader:
                if step >= 50:
                    break
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
                optimizer.zero_grad()
                out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
                loss = out.loss
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                step += 1
                if step in [1, 10, 25, 50]:
                    step_logs.append({"step": step, "loss": round(loss.item(), 4), "grad_norm": round(float(norm), 4)})
                    logger.info(f"  Step {step:02d}/50 | Loss: {loss.item():.4f} | Grad Norm: {norm:.4f}")

        post_acc, post_f1, post_cm, post_preds = evaluate(model, eval_enc, eval_labels, device=device)
        logger.info(f"[Rank {rank}] Post-Recovery: Acc={post_acc}% | F1={post_f1} | Preds={post_preds} | CM={post_cm}")

        diagnosis_results.append({
            "test": f"rank_{rank}",
            "rank": rank,
            "full_ft": False,
            "zero_shot_acc": zero_acc,
            "zero_shot_f1": zero_f1,
            "post_recovery_acc": post_acc,
            "post_recovery_f1": post_f1,
            "zero_preds": zero_preds,
            "post_preds": post_preds,
            "step_logs": step_logs
        })
        
        # Save incremental results to rank directory
        rank_dir = out_base / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        with open(rank_dir / "diagnosis.json", "w") as f:
            json.dump(diagnosis_results[-1], f, indent=2)

        del model
        torch.cuda.empty_cache()

    # 3. Test Full Fine-Tuning (Unfreeze All Parameters, lr=2e-5, 200 steps)
    logger.info(f"\n==================================================")
    logger.info(f"🧪 DIAGNOSIS: svd_llm_v2_attention with FULL FINE-TUNING (200 steps, lr=2e-5, rank=128)")
    logger.info(f"==================================================")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    inject_attention(model, "svd_llm_v2_attention", rank=128, copy_weights=True)
    
    zero_acc, zero_f1, zero_cm, zero_preds = evaluate(model, eval_enc, eval_labels, device=device)
    
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    step = 0
    step_logs_ft = []
    while step < 200:
        for b_ids, b_mask, b_labels in train_loader:
            if step >= 200:
                break
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = out.loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step in [1, 25, 50, 100, 150, 200]:
                step_logs_ft.append({"step": step, "loss": round(loss.item(), 4), "grad_norm": round(float(norm), 4)})
                logger.info(f"  Step {step:03d}/200 | Loss: {loss.item():.4f} | Grad Norm: {norm:.4f}")

    post_acc, post_f1, post_cm, post_preds = evaluate(model, eval_enc, eval_labels, device=device)
    logger.info(f"[Full FT] Post-Recovery: Acc={post_acc}% | F1={post_f1} | Preds={post_preds} | CM={post_cm}")

    diagnosis_results.append({
        "test": "full_fine_tuning_200steps",
        "rank": 128,
        "full_ft": True,
        "zero_shot_acc": zero_acc,
        "zero_shot_f1": zero_f1,
        "post_recovery_acc": post_acc,
        "post_recovery_f1": post_f1,
        "zero_preds": zero_preds,
        "post_preds": post_preds,
        "step_logs": step_logs_ft
    })

    ft_dir = out_base / "full_fine_tuning"
    ft_dir.mkdir(parents=True, exist_ok=True)
    with open(ft_dir / "diagnosis.json", "w") as f:
        json.dump(diagnosis_results[-1], f, indent=2)

    with open(out_base / "mrpc_majority_class_diagnosis.json", "w") as f:
        json.dump(diagnosis_results, f, indent=2)
        
    df = pd.DataFrame(diagnosis_results)
    df.to_csv(out_base / "mrpc_majority_class_diagnosis.csv", index=False)
    logger.info("Complete diagnosis saved to results/mrpc_test/mrpc_majority_class_diagnosis.json and CSV")

if __name__ == "__main__":
    main()
