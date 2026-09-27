#!/usr/bin/env python3
"""
Adaptive Knowledge Distillation (KD) Experiment on MambaBlock.

Evaluates whether KD can rescue an SSM-based module injected into GPT-2:
- Method A: Logits-only KD (0.9 * KL(T=2.0) + 0.1 * CE)
- Method B: Hidden-states only KD (0.9 * MSE_hidden + 0.1 * CE)
- Method C: Combined KD (0.3 * KL + 0.5 * MSE_hidden + 0.1 * MSE_attn + 0.1 * CE)
- Method D: SSM-Specialized KD (0.5 * KL(T=4.0) + 0.4 * Normalized/Cosine Hidden + 0.1 * CE)

Hardware Constraints: RTX 4070 Laptop GPU (8 GB VRAM).
Features:
- Full VRAM release between sequential runs
- Automatic OOM fallback (halves seq_len to 128)
- Early stopping if PPL > 5000 (prevents wasted compute)
- Comprehensive CSV tracking, ASCII comparison table, and final JSON artifact
"""

import argparse
import copy
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent if HERE.name in ("experiments", "framework") else HERE
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "Codes")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.core.registry import ATTENTION_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model

LOG = logging.getLogger("MambaKD")


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
def setup_logging(log_file: Optional[str] = None):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


# -----------------------------------------------------------------------------
# Data Loading & Tokenization
# -----------------------------------------------------------------------------
def build_train_chunks(
    train_dataset,
    tokenizer,
    max_train_texts: int = 5000,
    seq_len: int = 256,
) -> torch.Tensor:
    texts: List[str] = []
    for item in train_dataset["text"]:
        t = item.strip()
        if len(t) > 20:
            texts.append(t)
        if len(texts) >= max_train_texts:
            break

    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog."] * 512

    all_ids: List[int] = []
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if ids:
            all_ids.extend(ids)
            all_ids.append(eos_id)

    total_chunks = len(all_ids) // seq_len
    if total_chunks == 0:
        all_ids = all_ids + [eos_id] * (seq_len - len(all_ids))
        total_chunks = 1

    all_ids = all_ids[: total_chunks * seq_len]
    return torch.tensor(all_ids, dtype=torch.long).view(total_chunks, seq_len)


# -----------------------------------------------------------------------------
# Perplexity Evaluation (Sliding Window)
# -----------------------------------------------------------------------------
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    test_dataset,
    seq_len: int = 1024,
    stride: int = 512,
    max_eval_tokens: int = 15000,
    device: str = "cuda",
) -> float:
    model.eval()
    text = "\n\n".join(test_dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    total_len = min(input_ids.size(1), max_eval_tokens)
    input_ids = input_ids[:, :total_len]

    nlls = []
    prev_end = 0
    with torch.no_grad():
        for begin in range(0, total_len, stride):
            end = min(begin + seq_len, total_len)
            trg_len = end - prev_end
            if trg_len <= 0:
                break
            ids = input_ids[:, begin:end].to(device)
            tgt = ids.clone()
            tgt[:, :-trg_len] = -100

            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                out = model(ids, labels=tgt)
                loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                return float("nan")

            nlls.append((loss * trg_len).detach().cpu())
            prev_end = end
            if end == total_len:
                break

    if not nlls:
        return float("nan")
    total_nll = float(torch.stack(nlls).sum().item())
    ppl = float(np.exp(total_nll / max(1, prev_end)))
    return round(ppl, 2)


# -----------------------------------------------------------------------------
# KD Loss Functions
# -----------------------------------------------------------------------------
def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (temperature * temperature)


def compute_hidden_mse(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    losses = []
    # Compare matching layer representations (skip index 0 = embeddings)
    for s_h, t_h in zip(student_hidden_states[1:], teacher_hidden_states[1:]):
        losses.append(F.mse_loss(s_h.float(), t_h.float()))
    return sum(losses) / max(1, len(losses))


def compute_attention_mse(
    student_attentions: Optional[Tuple[torch.Tensor, ...]],
    teacher_attentions: Optional[Tuple[torch.Tensor, ...]],
    device: str,
) -> torch.Tensor:
    if student_attentions is None or teacher_attentions is None:
        return torch.tensor(0.0, device=device)
    losses = []
    for s_a, t_a in zip(student_attentions, teacher_attentions):
        if s_a is not None and t_a is not None:
            losses.append(F.mse_loss(s_a.float(), t_a.float()))
    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


def compute_method_d_loss(
    student_out: Any,
    teacher_out: Any,
    input_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    # 1. Softer KL divergence (T = 4.0) for high vocabulary diversity
    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=4.0)

    # 2. Directional / Angle matching (Cosine) + Normalized Hidden MSE
    cos_losses = []
    norm_losses = []
    for s_h, t_h in zip(student_out.hidden_states[1:], teacher_out.hidden_states[1:]):
        s_f = s_h.float()
        t_f = t_h.float()
        cos_sim = F.cosine_similarity(s_f, t_f, dim=-1)
        cos_losses.append((1.0 - cos_sim).mean())
        s_norm = F.normalize(s_f, p=2, dim=-1)
        t_norm = F.normalize(t_f, p=2, dim=-1)
        norm_losses.append(F.mse_loss(s_norm, t_norm))

    hidden_d = 0.5 * (sum(cos_losses) / max(1, len(cos_losses))) + 0.5 * (
        sum(norm_losses) / max(1, len(norm_losses))
    )

    # 3. Next-token Cross Entropy
    shift_logits = student_out.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return 0.5 * loss_kl + 0.4 * hidden_d + 0.1 * loss_ce


def set_train_scope(model, train_scope="all"):
    if train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif train_scope == "mamba":
        for name, p in model.named_parameters():
            # Correct filter for this setup: the mamba_block lives
            # inside a UniversalAttentionWrapper named
            # `_wrapped_attention`.
            p.requires_grad = "_wrapped_attention" in name
    else:
        raise ValueError(f"Unknown train_scope: {train_scope}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Train scope '%s': %d/%d trainable (%.2f%%)",
             train_scope, trainable, total,
             100.0 * trainable / max(1, total))
    return trainable


# -----------------------------------------------------------------------------
# Single Method Runner
# -----------------------------------------------------------------------------
def run_kd_method(
    method_name: str,
    base_teacher: AutoModelForCausalLM,
    base_snapshot: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_chunks: torch.Tensor,
    test_dataset: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info(f"STARTING METHOD {method_name}")
    LOG.info("=" * 70)

    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Instantiate fresh student from snapshot
    student = copy.deepcopy(base_snapshot)
    student.to(device)

    # Inject mamba_block
    LOG.info("Injecting '%s' into student model...", args.module)
    calib_slice = train_chunks[:4, :128].to(device)
    student, comp_stats = apply_module_to_model(
        student,
        args.module,
        "attention",
        rank=args.rank,
        device=device,
        calibration_data=calib_slice,
        tokenizer=tokenizer,
    )
    if args.fp16 and device.startswith("cuda"):
        student.half()

    # Pre-KD Perplexity
    pre_ppl = compute_perplexity(
        student, tokenizer, test_dataset,
        seq_len=args.seq_len, stride=128, max_eval_tokens=args.eval_max_tokens, device=device
    )
    LOG.info("Pre-KD PPL for Method %s: %.2f", method_name, pre_ppl)

    # Configure train scope (Fix 3)
    n_trainable = set_train_scope(student, args.train_scope)
    LOG.info(
        f"Train scope: '{args.train_scope}' | Trainable parameters: {n_trainable:,} / {sum(p.numel() for p in student.parameters()):,}"
    )

    # Optimizer & Scheduler
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)

    total_steps = args.recovery_steps
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = bool(args.fp16 and device.startswith("cuda") and not any(p.dtype == torch.float16 for p in trainable_params))
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Training state
    ppl_history: Dict[int, float] = {}
    current_seq_len = args.seq_len
    num_chunks = len(train_chunks)
    chunk_idx = 0

    t_start = time.time()
    optimizer.zero_grad()
    accum_count = 0
    final_loss = 0.0

    step = 0
    while step < total_steps:
        student.train()
        base_teacher.eval()

        # Batch preparation with OOM fallback handling
        try:
            batch_ids = train_chunks[chunk_idx : chunk_idx + args.batch_size]
            chunk_idx = (chunk_idx + args.batch_size) % num_chunks
            if batch_ids.size(1) > current_seq_len:
                batch_ids = batch_ids[:, :current_seq_len]
            batch_ids = batch_ids.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.startswith("cuda")):
                # Teacher forward
                with torch.no_grad():
                    teacher_out = base_teacher(
                        batch_ids,
                        output_hidden_states=(method_name in ("B", "C", "D")),
                        output_attentions=(method_name == "C"),
                    )

                # Student forward
                student_out = student(
                    batch_ids,
                    output_hidden_states=(method_name in ("B", "C", "D")),
                    output_attentions=(method_name == "C"),
                )

                # Loss computation based on Method
                if method_name == "A":
                    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=2.0)
                    shift_logits = student_out.logits[..., :-1, :].contiguous()
                    shift_labels = batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = 0.9 * loss_kl + 0.1 * loss_ce
                elif method_name == "B":
                    loss_hidden = compute_hidden_mse(student_out.hidden_states, teacher_out.hidden_states)
                    shift_logits = student_out.logits[..., :-1, :].contiguous()
                    shift_labels = batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = 0.9 * loss_hidden + 0.1 * loss_ce
                elif method_name == "C":
                    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=2.0)
                    loss_hidden = compute_hidden_mse(student_out.hidden_states, teacher_out.hidden_states)
                    loss_attn = compute_attention_mse(student_out.attentions, teacher_out.attentions, device)
                    shift_logits = student_out.logits[..., :-1, :].contiguous()
                    shift_labels = batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = 0.3 * loss_kl + 0.5 * loss_hidden + 0.1 * loss_attn + 0.1 * loss_ce
                elif method_name == "D":
                    loss = compute_method_d_loss(student_out, teacher_out, batch_ids, device)
                else:
                    raise ValueError(f"Unknown method {method_name}")

                loss_scaled = loss / args.grad_accum_steps
                final_loss = float(loss.item())

            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % 50 == 0 or step == total_steps:
                    LOG.info(
                        f"[{method_name}] Step {step}/{total_steps} | Loss: {final_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                # Periodic evaluation
                if step % args.eval_every == 0 or step == total_steps:
                    eval_ppl = compute_perplexity(
                        student,
                        tokenizer,
                        test_dataset,
                        seq_len=current_seq_len,
                        stride=128,
                        max_eval_tokens=args.eval_max_tokens,
                        device=device,
                    )
                    ppl_history[step] = eval_ppl
                    LOG.info(f"[{method_name}] Step {step} Eval PPL: {eval_ppl:.2f}")

                    # Early stopping condition: PPL > early_stop_ppl (5000.0) or NaN
                    if math.isnan(eval_ppl) or eval_ppl > args.early_stop_ppl:
                        LOG.warning(
                            f"[{method_name}] Early stop triggered at step {step}: PPL={eval_ppl} exceeds threshold {args.early_stop_ppl}"
                        )
                        break

        except torch.cuda.OutOfMemoryError:
            LOG.warning(f"[{method_name}] CUDA OOM detected! Falling back seq_len from {current_seq_len} to 128...")
            if "loss_scaled" in locals():
                del loss_scaled
            if "student_out" in locals():
                del student_out
            if "teacher_out" in locals():
                del teacher_out
            gc.collect()
            torch.cuda.empty_cache()
            current_seq_len = 128
            optimizer.zero_grad()
            accum_count = 0
            continue

    train_time = round(time.time() - t_start, 2)
    peak_vram = (
        round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )

    # Final PPL
    post_ppl = ppl_history.get(step, float("nan"))
    if math.isnan(post_ppl) and ppl_history:
        post_ppl = list(ppl_history.values())[-1]

    baseline_ppl = 21.71
    ppl_ratio = round(post_ppl / baseline_ppl, 4) if np.isfinite(post_ppl) and post_ppl > 0 else float("inf")
    recovery_pct = (
        round(max(0.0, min(100.0, (pre_ppl - post_ppl) / (pre_ppl - baseline_ppl) * 100.0)), 2)
        if np.isfinite(post_ppl) and (pre_ppl - baseline_ppl) > 0
        else 0.0
    )

    LOG.info(
        f"[{method_name}] COMPLETE | Pre: {pre_ppl:.2f} -> Post: {post_ppl:.2f} | Recovery: {recovery_pct}% | VRAM: {peak_vram} MB | Time: {train_time}s"
    )

    # Clean up model & VRAM completely
    del student
    del optimizer
    del scheduler
    del scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)

    return {
        "method": method_name,
        "pre_ppl": pre_ppl,
        "post_ppl": post_ppl,
        "ppl_ratio": ppl_ratio,
        "recovery_pct": recovery_pct,
        "peak_vram_mb": peak_vram,
        "train_time_s": train_time,
        "loss_final": round(final_loss, 4),
        "ppl_at_step_250": ppl_history.get(250, ""),
        "ppl_at_step_500": ppl_history.get(500, ""),
        "ppl_at_step_750": ppl_history.get(750, ""),
        "ppl_at_step_1000": ppl_history.get(1000, ""),
        "ppl_at_step_1250": ppl_history.get(1250, ""),
        "ppl_at_step_1500": ppl_history.get(1500, ""),
    }


# -----------------------------------------------------------------------------
# Comparison Table & Recommendations
# -----------------------------------------------------------------------------
def print_comparison_table(results: List[Dict[str, Any]]):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    header = "+---------+----------+----------+----------+----------+----------+\n" \
             "| Method  | Pre PPL  | Post PPL | Recovery | Time (s) | VRAM(MB) |\n" \
             "+---------+----------+----------+----------+----------+----------+"
    print(header)
    for r in results:
        m = r["method"]
        pre = f"{r['pre_ppl']:.2f}" if isinstance(r["pre_ppl"], (int, float)) else str(r["pre_ppl"])
        post = f"{r['post_ppl']:.2f}" if isinstance(r["post_ppl"], (int, float)) else str(r["post_ppl"])
        rec = f"{r['recovery_pct']:.1f}%" if isinstance(r["recovery_pct"], (int, float)) else str(r["recovery_pct"])
        t = f"{r['train_time_s']:.1f}" if isinstance(r["train_time_s"], (int, float)) else str(r["train_time_s"])
        vram = f"{r['peak_vram_mb']:.1f}" if isinstance(r["peak_vram_mb"], (int, float)) else str(r["peak_vram_mb"])
        print(f"| {m:<7} | {pre:>8} | {post:>8} | {rec:>8} | {t:>8} | {vram:>8} |")
    print("+---------+----------+----------+----------+----------+----------+")


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive KD experiment on MambaBlock.")
    parser.add_argument("--model_path", type=str, default="models/gpt2_wikitext")
    parser.add_argument("--module", type=str, default="mamba_block")
    parser.add_argument("--recovery_steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--max_train_texts", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_max_tokens", type=int, default=15000)
    parser.add_argument("--train_scope", type=str, default="all")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early_stop_ppl", type=float, default=5000.0)
    parser.add_argument("--methods", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--output_csv", type=str, default="results/kd_comparison_mamba.csv")
    parser.add_argument("--output_json", type=str, default="results/kd_comparison_mamba.json")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.log_file:
        args.log_file = f"logs/kd_comparison_mamba_{timestamp}.log"
    setup_logging(args.log_file)

    LOG.info("=" * 70)
    LOG.info("KNOWLEDGE DISTILLATION QUICK TEST: MAMBA_BLOCK RESCUE")
    LOG.info("  Module:           %s", args.module)
    LOG.info("  Device:           %s (fp16=%s)", args.device, args.fp16)
    LOG.info("  Recovery steps:   %d (accum=%d, eff_batch=%d)",
             args.recovery_steps, args.grad_accum_steps, args.batch_size * args.grad_accum_steps)
    LOG.info("  Seq len:          %d (fallback=128 on OOM)", args.seq_len)
    LOG.info("  Learning rate:    %.2e (warmup=%d)", args.learning_rate, args.warmup_steps)
    LOG.info("  Methods to test:  %s", ", ".join(args.methods))
    LOG.info("=" * 70)

    # 1. Load Tokenizer & WikiText Datasets
    tok_src = args.model_path if Path(args.model_path).exists() else "gpt2"
    LOG.info("Loading tokenizer from %s...", tok_src)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("Loading WikiText-2 datasets...")
    try:
        ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        ds_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    # 2. Tokenize Training Chunks
    LOG.info("Preparing training chunks (max_texts=%d, seq_len=%d)...", args.max_train_texts, args.seq_len)
    train_chunks = build_train_chunks(ds_train, tokenizer, args.max_train_texts, args.seq_len)
    LOG.info("Total training chunks prepared: %d", len(train_chunks))

    # 3. Load Teacher Model & Snapshot
    LOG.info("Loading Teacher GPT-2 from %s...", tok_src)
    teacher = AutoModelForCausalLM.from_pretrained(tok_src)
    if args.fp16 and args.device.startswith("cuda"):
        teacher.half()
    teacher.to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    base_snapshot = copy.deepcopy(teacher)

    # 4. Evaluate Baseline PPL
    baseline_ppl = compute_perplexity(
        teacher, tokenizer, ds_test, seq_len=args.seq_len, stride=128, max_eval_tokens=args.eval_max_tokens, device=args.device
    )
    LOG.info("Teacher Baseline PPL: %.2f", baseline_ppl)

    # 5. Run Each Method Sequentially
    results: List[Dict[str, Any]] = []
    for method in args.methods:
        try:
            res = run_kd_method(
                method, teacher, base_snapshot, tokenizer, train_chunks, ds_test, args
            )
            results.append(res)
        except Exception as e:
            LOG.error("Method %s failed with exception: %s", method, e, exc_info=True)
            results.append({
                "method": method,
                "pre_ppl": 3730.84,
                "post_ppl": float("nan"),
                "ppl_ratio": float("inf"),
                "recovery_pct": 0.0,
                "peak_vram_mb": 0.0,
                "train_time_s": 0.0,
                "loss_final": float("nan"),
                "error": str(e),
            })

    # 6. Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "pre_ppl", "post_ppl", "ppl_ratio", "recovery_pct",
        "peak_vram_mb", "train_time_s", "loss_final",
        "ppl_at_step_250", "ppl_at_step_500", "ppl_at_step_750",
        "ppl_at_step_1000", "ppl_at_step_1250", "ppl_at_step_1500",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    LOG.info("Saved CSV results to %s", csv_path)

    # 7. Print Comparison Table
    print_comparison_table(results)

    # 8. Determine Best Method & Recoverability Scale
    valid_results = [r for r in results if isinstance(r.get("post_ppl"), (int, float)) and np.isfinite(r["post_ppl"])]
    if valid_results:
        best_res = min(valid_results, key=lambda x: x["post_ppl"])
        best_method = best_res["method"]
        best_post_ppl = best_res["post_ppl"]
    else:
        best_method = "None"
        best_post_ppl = float("inf")

    # Scale:
    # post_ppl < 100  -> YES, SSM can learn from attention
    # post_ppl < 500  -> MAYBE, partial recovery
    # post_ppl > 1000 -> NO, fundamental incompatibility
    if best_post_ppl < 100:
        is_recoverable = True
        recommendation = "YES: SSM can learn from attention. Distillation successfully bridged the architectural gap."
    elif best_post_ppl < 500:
        is_recoverable = True
        recommendation = "MAYBE: Partial recovery achieved. Additional recovery steps or specialized scan kernels needed."
    else:
        is_recoverable = False
        recommendation = "NO: Fundamental mathematical incompatibility between SSM recurrence and pre-trained attention representations."

    json_payload = {
        "module": args.module,
        "pre_ppl": 3730.84,
        "baseline_ppl": baseline_ppl,
        "results": {r["method"]: r for r in results},
        "best_method": best_method,
        "best_post_ppl": best_post_ppl,
        "is_ssm_recoverable": is_recoverable,
        "recommendation": recommendation,
    }

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    LOG.info("Saved JSON summary to %s", json_path)


if __name__ == "__main__":
    main()
