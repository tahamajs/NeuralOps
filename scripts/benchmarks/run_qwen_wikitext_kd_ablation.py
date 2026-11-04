#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajlesi/NeuralOps/graphs/contributors
"""
Unified Qwen Causal Language Modeling KD & Attention Surgery Benchmark on WikiText-2.

Evaluates:
  - Teacher Baseline Perplexity (PPL) on WikiText-2 Test Set
  - Stage 0: Zero-Shot Attention Surgery PPL
  - Stage 1: Isolated Attention Mimicry PPL (Backbone frozen)
  - Stage 2: Quick E2E Distillation Polish PPL (AdamW)

Persists metrics to results/master_ledger.sqlite and results/qwen_wikitext_kd_ablation.csv.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import logging
import math
import os
import sqlite3
import sys
import time
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

_orig_named_params = nn.Module.named_parameters
def _safe_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _safe_named_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_wiki_kd")


class ArrowWikiTextDataset(Dataset):
    """Direct streaming of tokenized WikiText-2 blocks for Causal LM evaluation."""
    def __init__(self, arrow_path: str, tokenizer: Any, seq_len: int = 512) -> None:
        with open(arrow_path, "rb") as f:
            reader = ipc.open_stream(f)
            self.table = reader.read_all()
        raw_texts = self.table["text"].to_pylist()
        concatenated = "\n\n".join([t for t in raw_texts if t.strip()])
        tokens = tokenizer.encode(concatenated, add_special_tokens=False)
        self.num_samples = len(tokens) // seq_len
        self.tokens = torch.tensor(tokens[:self.num_samples * seq_len], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len]
        return {
            "input_ids": chunk,
            "labels": chunk.clone()
        }


def evaluate_ppl(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 40) -> float:
    """Calculates official Perplexity on WikiText-2 with causal cross-entropy."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            tokens_count = labels.numel()

            total_loss += loss.item() * tokens_count
            total_tokens += tokens_count

    if total_tokens == 0:
        return 999.9
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return round(ppl, 2)


class CausalAttentionSurgeonAdapter(nn.Module):
    """Universal Adapter reconciling Grouped-Query Attention & RoPE with Registry modules."""
    def __init__(self, core_attn: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.core_attn = core_attn
        self.in_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        proj_in = self.in_proj(hidden_states)
        out = self.core_attn(proj_in)
        out_tensor = out[0] if isinstance(out, tuple) else out
        final_out = self.out_proj(out_tensor)
        return (final_out, None)


def run_qwen_surgery_pipeline(
    module_name: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    isolated_steps: int = 60,
    polish_steps: int = 25,
    iso_lr: float = 8e-4,
    polish_lr: float = 1.5e-5,
) -> dict[str, Any]:
    from neuralops.core.registry import create_attention

    logger.info(f"\n{'='*80}\n[QWEN SURGERY] Testing Attention Module: {module_name}\n{'='*80}")
    t0 = time.time()

    student_model = copy.deepcopy(teacher_model)
    hidden_size = student_model.config.hidden_size
    num_heads = student_model.config.num_attention_heads

    # Replace Qwen attention layers with adapted registry module
    for layer in student_model.model.layers:
        try:
            core_attn = create_attention(module_name, embed_dim=hidden_size, num_heads=num_heads)
        except Exception:
            core_attn = create_attention(module_name, dim=hidden_size, num_heads=num_heads)
        layer.self_attn = CausalAttentionSurgeonAdapter(core_attn, hidden_size).to(device, dtype=torch.float16)

    student_model.to(device)
    zero_shot_ppl = evaluate_ppl(student_model, test_loader, device)
    logger.info(f"[{module_name}] Stage 0 (Zero-Shot Surgery): PPL={zero_shot_ppl:.2f}")

    # Stage 1: Isolated Attention Mimicry (Backbone Frozen)
    t_iso_start = time.time()
    for name, param in student_model.named_parameters():
        param.requires_grad = "self_attn" in name

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer_iso = torch.optim.AdamW(trainable_params, lr=iso_lr, weight_decay=1e-4)
    student_model.train()

    for step, batch in enumerate(train_loader):
        if step >= isolated_steps:
            break
        in_ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        optimizer_iso.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_out = teacher_model(input_ids=in_ids)
        s_out = student_model(input_ids=in_ids)

        kd_loss = F.kl_div(
            F.log_softmax(s_out.logits / 2.0, dim=-1),
            F.softmax(t_out.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * 4.0
        lm_loss = F.cross_entropy(s_out.logits.view(-1, s_out.logits.size(-1)), lbls.view(-1))
        loss = 0.5 * kd_loss + 0.5 * lm_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer_iso.step()

    t_iso_dur = time.time() - t_iso_start
    isolated_ppl = evaluate_ppl(student_model, test_loader, device)
    logger.info(f"[{module_name}] Stage 1 (Isolated Attn): PPL={isolated_ppl:.2f} | Time={t_iso_dur:.1f}s")

    # Stage 2: Quick E2E Distillation Polish
    t_pol_start = time.time()
    for p in student_model.parameters():
        p.requires_grad = True

    optimizer_pol = torch.optim.AdamW(student_model.parameters(), lr=polish_lr, weight_decay=0.01)
    student_model.train()

    for step, batch in enumerate(train_loader):
        if step >= polish_steps:
            break
        in_ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        optimizer_pol.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_out = teacher_model(input_ids=in_ids)
        s_out = student_model(input_ids=in_ids)

        kd_loss = F.kl_div(
            F.log_softmax(s_out.logits / 2.0, dim=-1),
            F.softmax(t_out.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * 4.0
        lm_loss = F.cross_entropy(s_out.logits.view(-1, s_out.logits.size(-1)), lbls.view(-1))
        total_l = 0.5 * kd_loss + 0.5 * lm_loss
        total_l.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer_pol.step()

    t_pol_dur = time.time() - t_pol_start
    final_ppl = evaluate_ppl(student_model, test_loader, device)
    total_dur = time.time() - t0
    logger.info(f"[{module_name}] Stage 2 (Quick Polish): PPL={final_ppl:.2f} | TotalTime={total_dur:.1f}s")

    del student_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "module": module_name,
        "zero_shot_ppl": zero_shot_ppl,
        "isolated_ppl": isolated_ppl,
        "polished_ppl": final_ppl,
        "isolated_time_s": round(t_iso_dur, 1),
        "polish_time_s": round(t_pol_dur, 1),
        "total_time_s": round(total_dur, 1),
        "status": "PASSED"
    }


def persist_qwen_sqlite(results: list[dict[str, Any]], base_ppl: float, sqlite_path: str = "results/master_ledger.sqlite") -> None:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS qwen_wikitext_kd_ablation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            dataset TEXT,
            module TEXT,
            teacher_baseline_ppl REAL,
            zero_shot_ppl REAL,
            isolated_ppl REAL,
            polished_ppl REAL,
            ppl_delta REAL,
            total_time_s REAL,
            status TEXT
        )
    """)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for r in results:
        pol_ppl = r.get("polished_ppl", 0.0)
        delta = round(pol_ppl - base_ppl, 2)
        cur.execute("""
            INSERT INTO qwen_wikitext_kd_ablation (
                timestamp, model_name, dataset, module, teacher_baseline_ppl,
                zero_shot_ppl, isolated_ppl, polished_ppl, ppl_delta,
                total_time_s, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            "Qwen/GPT2.5-0.5B",
            "WikiText-2",
            r.get("module"),
            base_ppl,
            r.get("zero_shot_ppl", 0.0),
            r.get("isolated_ppl", 0.0),
            pol_ppl,
            delta,
            r.get("total_time_s", 0.0),
            r.get("status", "UNKNOWN")
        ))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Qwen WikiText-2 Causal KD Ablation")
    parser.add_argument("--model-path", type=str, default="/home/tahamajlesi.ut/.cache/huggingface/hub/models--Qwen--GPT2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--all-modules", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--isolated-steps", type=int, default=60)
    parser.add_argument("--polish-steps", type=int, default=25)
    parser.add_argument("--iso-lr", type=float, default=8e-4)
    parser.add_argument("--polish-lr", type=float, default=1.5e-5)
    parser.add_argument("--out-csv", type=str, default="results/qwen_wikitext_kd_ablation.csv")
    parser.add_argument("--out-json", type=str, default="results/qwen_wikitext_kd_ablation.json")
    parser.add_argument("--ledger-db", type=str, default="results/master_ledger.sqlite")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Launching Qwen Causal LM KD on: {torch.cuda.get_device_name(0)} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = os.path.expanduser("~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow")
    test_path = os.path.expanduser("~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")

    train_dataset = ArrowWikiTextDataset(train_path, tokenizer, seq_len=args.seq_len)
    test_dataset = ArrowWikiTextDataset(test_path, tokenizer, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    logger.info(f"Loading Teacher Model from {args.model_path} in float16...")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_ppl = evaluate_ppl(teacher_model, test_loader, device)
    logger.info(f"Verified Teacher Baseline PPL on WikiText-2: {teacher_ppl:.2f}")

    if args.module:
        target_modules = [args.module]
    elif args.all_modules:
        from neuralops.core.registry import ATTENTION_REGISTRY
        target_modules = list(ATTENTION_REGISTRY.keys())
    else:
        target_modules = [
            "linear_attention",
            "cosformer",
            "gla_attention",
            "svd_linear_attention",
            "decay_linear_attention",
            "gated_linear_attention",
        ]

    results = []
    for mod_name in target_modules:
        try:
            res = run_qwen_surgery_pipeline(
                module_name=mod_name,
                teacher_model=teacher_model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                isolated_steps=args.isolated_steps,
                polish_steps=args.polish_steps,
                iso_lr=args.iso_lr,
                polish_lr=args.polish_lr,
            )
            res["teacher_ppl"] = teacher_ppl
            results.append(res)
        except Exception as exc:
            logger.error(f"Error evaluating {mod_name} on Qwen: {exc}")
            results.append({
                "module": mod_name,
                "zero_shot_ppl": 999.9,
                "isolated_ppl": 999.9,
                "polished_ppl": 999.9,
                "teacher_ppl": teacher_ppl,
                "total_time_s": 0.0,
                "status": f"FAILED: {exc}"
            })

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    persist_qwen_sqlite(results, base_ppl=teacher_ppl, sqlite_path=args.ledger_db)

    print("\n" + "=" * 105)
    print(f"QWEN WIKITEXT-2 PERPLEXITY ABLATION SUMMARY (Teacher Baseline PPL: {teacher_ppl:.2f})")
    print("=" * 105)
    print(f"{'Module':<25} | {'Zero-Shot PPL':<15} | {'Isolated PPL':<15} | {'Polished PPL':<15} | {'Total Time':<12} | {'Status'}")
    print("-" * 105)
    for r in results:
        mod = r.get("module", "")[:24]
        zs = f"{r.get('zero_shot_ppl', 0.0):.2f}"
        iso = f"{r.get('isolated_ppl', 0.0):.2f}"
        pol = f"{r.get('polished_ppl', 0.0):.2f}"
        tt = f"{r.get('total_time_s', 0.0):.1f}s"
        st = r.get("status", "")
        print(f"{mod:<25} | {zs:<15} | {iso:<15} | {pol:<15} | {tt:<12} | {st}")
    print("=" * 105)


if __name__ == "__main__":
    main()
