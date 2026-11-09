#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 1 — Measure vanilla BERT-base-uncased perplexity on WikiText-2.

Uses Masked Language Modeling (MLM) loss averaged over the validation split.
PPL = exp(mean_cross_entropy_loss_over_all_tokens).

Output JSON: {"ppl_vanilla_bert": float, "loss": float, "num_batches": int,
              "num_tokens": int, "timestamp": str, "model": str, "dataset": str}
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bert_zoo_baseline_ppl")

MASK_PROB = 0.15


def build_mlm_dataset(tokenizer, dataset_name: str, split: str, seq_len: int):
    """Tokenize WikiText and return a simple tensor dataset for MLM PPL."""
    from datasets import load_dataset
    from torch.utils.data import Dataset

    raw = load_dataset("wikitext", dataset_name, split=split)
    text = "\n\n".join([t for t in raw["text"] if t.strip()])
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"].squeeze(0)

    # Chunk into seq_len blocks
    n = (tokens.size(0) // seq_len) * seq_len
    tokens = tokens[:n].view(-1, seq_len)

    class WikiMLMDataset(Dataset):
        def __init__(self, chunks, vocab_size, mask_id, cls_id, sep_id, pad_id, seed: int = 42):
            self.chunks = chunks
            self.vocab_size = vocab_size
            self.mask_id = mask_id
            self.cls_id = cls_id
            self.sep_id = sep_id
            self.pad_id = pad_id
            self.seed = seed

        def __len__(self):
            return self.chunks.size(0)

        def __getitem__(self, idx):
            ids = self.chunks[idx].clone()
            # Prepend CLS, append SEP (truncate to seq_len)
            ids = torch.cat([
                torch.tensor([self.cls_id]),
                ids[: len(ids) - 2],
                torch.tensor([self.sep_id]),
            ])
            labels = ids.clone()
            # Deterministic 15% masking with per-index seed
            g = torch.Generator().manual_seed(self.seed + idx)
            prob = torch.rand(len(ids), generator=g)
            maskable = (ids != self.cls_id) & (ids != self.sep_id)
            mask = (prob < MASK_PROB) & maskable

            # 80% MASK, 10% random, 10% keep
            rand_sub = torch.rand(len(ids), generator=g)
            mask_token_mask = mask & (rand_sub < 0.8)
            random_token_mask = mask & (rand_sub >= 0.8) & (rand_sub < 0.9)

            ids[mask_token_mask] = self.mask_id
            if random_token_mask.any():
                random_words = torch.randint(0, self.vocab_size, (random_token_mask.sum().item(),), generator=g)
                ids[random_token_mask] = random_words

            # Non-masked positions -> -100 so they don't contribute to loss/accuracy
            labels[~mask] = -100
            attn_mask = torch.ones(len(ids), dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": attn_mask,
                "labels": labels,
            }

    return WikiMLMDataset(
        tokens,
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_token_id,
        cls_id=tokenizer.cls_token_id,
        sep_id=tokenizer.sep_token_id,
        pad_id=tokenizer.pad_token_id,
        seed=42,
    )


def compute_ppl(model, dataloader, device) -> tuple[float, float, float, int, int]:
    """
    Returns (ppl, avg_loss, accuracy_pct, num_batches, num_tokens).
    Uses sum-of-losses / total_masked_tokens for proper per-token PPL.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            mask = labels != -100
            n_masked = mask.sum().item()
            if n_masked == 0:
                continue

            total_loss += out.loss.item() * n_masked
            total_tokens += n_masked

            preds = out.logits.argmax(dim=-1)
            correct_tokens += (preds[mask] == labels[mask]).sum().item()
            num_batches += 1

    if total_tokens == 0:
        raise RuntimeError("No masked tokens found — check dataset construction.")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc_pct = (correct_tokens / total_tokens) * 100.0
    return ppl, avg_loss, acc_pct, num_batches, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vanilla BERT PPL on WikiText-2")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default="results/bert_attention_zoo/baseline_ppl.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available.")
        sys.exit(3)

    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    logger.info("Building WikiText MLM dataset...")
    dataset = build_mlm_dataset(tokenizer, args.dataset, args.split, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.info(f"Dataset: {len(dataset)} chunks of seq_len={args.seq_len}")

    logger.info("Computing PPL and Masked Accuracy...")
    ppl, avg_loss, acc_pct, num_batches, num_tokens = compute_ppl(model, loader, device)
    logger.info(
        f"Vanilla BERT PPL: {ppl:.4f} | Loss: {avg_loss:.4f} | "
        f"Acc: {acc_pct:.2f}% | Batches: {num_batches} | Tokens: {num_tokens}"
    )

    result = {
        "ppl_vanilla_bert": round(ppl, 4),
        "acc_vanilla_bert": round(acc_pct, 2),
        "loss": round(avg_loss, 6),
        "num_batches": num_batches,
        "num_tokens": num_tokens,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Baseline PPL written to: {args.output}")


if __name__ == "__main__":
    main()
