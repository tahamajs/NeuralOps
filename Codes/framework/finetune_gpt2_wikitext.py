#!/usr/bin/env python3
"""
Fine-tune GPT-2 Small on WikiText-2 (raw) for causal language modeling.

Usage:
    python finetune_gpt2_wikitext.py --epochs 2 --batch_size 4 --lr 5e-5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 small on WikiText-2 raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Base model to fine-tune (HuggingFace ID or local path)")
    parser.add_argument("--output_dir", type=str, default="models/gpt2_wikitext",
                        help="Directory to save fine-tuned model checkpoint")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1",
                        help="WikiText config name")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Block size / sequence length for language modeling")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding-window evaluation")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--upload_to_hf", action="store_true", default=False,
                        help="Upload fine-tuned model to HuggingFace hub")
    parser.add_argument("--hf_repo_id", type=str, default="Hooshaai/gpt2-wikitext",
                        help="HuggingFace destination repository ID")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_perplexity(model, tokenizer, dataset, seq_len=1024, stride=512, device="cuda"):
    """
    Standard HuggingFace sliding-window perplexity calculation.
    PPL = exp( sum(NLL * target_length) / total_target_tokens )
    """
    model.eval()
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = min(seq_len, encodings.input_ids.size(1))
    
    nlls = []
    prev_end_loc = 0
    total_tokens = encodings.input_ids.size(1)

    with torch.no_grad():
        for begin_loc in range(0, total_tokens, stride):
            end_loc = min(begin_loc + seq_len, total_tokens)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.detach().cpu())

            prev_end_loc = end_loc
            if end_loc == total_tokens:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return float(ppl.item())


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Fine-tuning {args.model_name_or_path} on {args.dataset_name} using device: {device}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    # Load WikiText-2
    try:
        raw_datasets = load_dataset("wikitext", args.dataset_name)
    except Exception:
        raw_datasets = load_dataset("Salesforce/wikitext", args.dataset_name)

    # Tokenize and group texts into blocks of seq_len
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing WikiText-2",
    )

    block_size = args.seq_len

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping into chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    val_dataset = raw_datasets["validation"]

    # DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Initial Zero-Shot Perplexity
    print("[EVAL] Measuring baseline validation perplexity before fine-tuning...")
    initial_ppl = compute_perplexity(
        model, tokenizer, val_dataset, seq_len=args.seq_len, stride=args.stride, device=device
    )
    print(f"[EVAL] Pre-training Baseline Perplexity: {initial_ppl:.2f}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    # Training Loop
    print(f"[TRAIN] Starting training: {args.epochs} epochs ({total_steps} steps, batch_size={args.batch_size})")
    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0 or global_step == total_steps:
                avg_loss = epoch_loss / (step + 1)
                cur_ppl = math.exp(min(avg_loss, 20.0))
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | "
                      f"Loss: {avg_loss:.4f} | Est PPL: {cur_ppl:.2f} | Time: {elapsed:.1f}s")

    # Final Validation Perplexity
    print("[EVAL] Measuring final validation perplexity...")
    final_ppl = compute_perplexity(
        model, tokenizer, val_dataset, seq_len=args.seq_len, stride=args.stride, device=device
    )
    print(f"[EVAL] Final Fine-Tuned Validation Perplexity: {final_ppl:.2f}")

    # Save Artifacts
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Saving model and tokenizer to {out_path}...")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Save metrics metadata
    import json
    metadata = {
        "model": args.model_name_or_path,
        "dataset": args.dataset_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "initial_ppl": initial_ppl,
        "final_ppl": final_ppl,
        "train_time_s": time.time() - t0,
    }
    (out_path / "finetune_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] Checkpoint saved successfully. Metadata: {metadata}")

    if args.upload_to_hf:
        print(f"[HF] Uploading to HuggingFace Hub: {args.hf_repo_id}...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=args.hf_repo_id,
            commit_message=f"Upload fine-tuned GPT-2 on WikiText-2 (PPL={final_ppl:.2f})",
        )
        print("[HF] Upload completed.")


if __name__ == "__main__":
    main()
