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
- Teacher output preprocessing cache (5-10x speedup across iterations)
- Full VRAM release between sequential runs
- Automatic OOM fallback (halves seq_len to 128)
- Early stopping if PPL > 5000 (prevents wasted compute)
- Comprehensive CSV tracking, ASCII comparison table, and final JSON artifact
"""

import argparse
import copy
import csv
import gc
import hashlib
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
            if trg_len < (end - begin):
                tgt[:, :-trg_len] = -100

            with torch.amp.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
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
    """
    Configure trainable parameters.

    - 'all'                 : train everything
    - 'mamba'/'attention'/'module' : train ONLY parameters inside
                                    the attention submodule, EXCEPT the
                                    original GPT-2 c_attn/c_proj fused
                                    projections. This works for ANY
                                    injected attention module, whether it
                                    uses UniversalAttentionWrapper or
                                    injects directly (q_proj, k_proj, ...).
    """
    if train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif train_scope in ("mamba", "attention", "module"):
        for name, p in model.named_parameters():
            # Must be inside a transformer block's attention submodule
            if ".attn." not in name:
                p.requires_grad = False
                continue

            # Within .attn., inspect the immediate submodule name
            sub = name.split(".attn.", 1)[-1]

            # Skip original GPT-2 fused projections
            if sub.startswith(("c_attn", "c_proj")):
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown train_scope: {train_scope}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Train scope '%s': %d/%d trainable (%.2f%%)",
             train_scope, trainable, total,
             100.0 * trainable / max(1, total))
    return trainable


# -----------------------------------------------------------------------------
# Teacher Output Preprocessing Cache
# -----------------------------------------------------------------------------
class CachedTeacherOutput:
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        attentions: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


def get_cache_hash(
    model_path: str,
    tokenizer_name: str,
    seq_len: int,
    num_chunks: int,
    methods: List[str],
    include_attn: bool,
) -> str:
    cache_key_data = {
        "model_path": str(model_path),
        "tokenizer_name": str(tokenizer_name),
        "seq_len": int(seq_len),
        "num_chunks": int(num_chunks),
        "methods": sorted(list(set(methods))),
        "include_attn": bool(include_attn),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()[:12]


def is_cache_valid(
    cache_dir: Path,
    expected_chunks: int,
    expected_seq_len: int,
    methods: List[str],
    include_attn: bool = False,
) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("n_chunks") != expected_chunks:
            return False
        if manifest.get("seq_len") != expected_seq_len:
            return False
        methods_set = set(methods)
        need_logits = any(m in methods_set for m in ("A", "C", "D"))
        need_hidden = any(m in methods_set for m in ("B", "C", "D"))
        need_attn = include_attn and ("C" in methods_set)
        if need_logits and not (manifest.get("has_logits") and (cache_dir / (manifest.get("logits_file") or "")).exists()):
            return False
        if need_hidden and not (manifest.get("has_hidden") and (cache_dir / (manifest.get("hidden_file") or "")).exists()):
            return False
        if need_attn and not (manifest.get("has_attn") and (cache_dir / (manifest.get("attn_file") or "")).exists()):
            return False
        return True
    except Exception:
        return False


def build_teacher_cache(
    teacher: nn.Module,
    tokenizer,
    train_chunks: torch.Tensor,
    cache_dir: Path,
    methods: List[str],
    device: str,
    seq_len: int,
    include_attn: bool = False,
    max_train_texts: int = 1000,
) -> Path:
    """
    Runs teacher forward on every chunk, saves outputs to cache_dir.
    Returns the path to the cache manifest.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = len(train_chunks)

    methods_set = set(methods)
    need_logits = any(m in methods_set for m in ("A", "C", "D"))
    need_hidden = any(m in methods_set for m in ("B", "C", "D"))
    need_attn = include_attn and ("C" in methods_set)

    LOG.info(
        "Building teacher cache for %d chunks (max_texts=%d, seq_len=%d)...",
        n_chunks, max_train_texts, seq_len
    )
    t0 = time.time()

    teacher.eval()
    teacher.to(device)

    eval_batch_size = min(8, n_chunks)
    all_logits: List[torch.Tensor] = []
    all_hidden_batches: List[List[torch.Tensor]] = []
    all_attn_batches: List[List[torch.Tensor]] = []

    with torch.no_grad():
        for start_idx in range(0, n_chunks, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_chunks)
            chunk_batch = train_chunks[start_idx:end_idx]
            if chunk_batch.size(1) > seq_len:
                chunk_batch = chunk_batch[:, :seq_len]
            chunk_batch = chunk_batch.to(device)

            with torch.amp.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                out = teacher(
                    chunk_batch,
                    output_hidden_states=need_hidden,
                    output_attentions=need_attn,
                )

            if need_logits:
                all_logits.append(out.logits.detach().to(torch.float16).cpu())

            if need_hidden and out.hidden_states is not None:
                batch_layers = [h.detach().to(torch.float16).cpu() for h in out.hidden_states]
                all_hidden_batches.append(batch_layers)

            if need_attn and out.attentions is not None:
                batch_attns = [a.detach().to(torch.float16).cpu() for a in out.attentions]
                all_attn_batches.append(batch_attns)

    manifest: Dict[str, Any] = {
        "n_chunks": n_chunks,
        "seq_len": seq_len,
        "vocab_size": len(tokenizer),
        "has_logits": need_logits,
        "has_hidden": need_hidden,
        "has_attn": need_attn,
        "logits_file": None,
        "hidden_file": None,
        "attn_file": None,
        "logits_size_bytes": 0,
        "hidden_size_bytes": 0,
        "attn_size_bytes": 0,
        "build_time_s": 0.0,
    }

    total_size_bytes = 0
    logits_gb = 0.0
    if need_logits and all_logits:
        logits_tensor = torch.cat(all_logits, dim=0)
        logits_path = cache_dir / "teacher_logits.pt"
        torch.save(logits_tensor, logits_path, _use_new_zipfile_serialization=True)
        sz = logits_path.stat().st_size
        manifest["logits_file"] = "teacher_logits.pt"
        manifest["logits_size_bytes"] = sz
        total_size_bytes += sz
        logits_gb = sz / 1e9
        del logits_tensor, all_logits

    if need_hidden and all_hidden_batches:
        num_layers = len(all_hidden_batches[0])
        stacked_layers = []
        for l in range(num_layers):
            stacked_layers.append(torch.cat([b[l] for b in all_hidden_batches], dim=0))
        hidden_tensor = torch.stack(stacked_layers, dim=0)
        hidden_path = cache_dir / "teacher_hidden.pt"
        torch.save(hidden_tensor, hidden_path, _use_new_zipfile_serialization=True)
        sz = hidden_path.stat().st_size
        manifest["hidden_file"] = "teacher_hidden.pt"
        manifest["hidden_size_bytes"] = sz
        total_size_bytes += sz
        del hidden_tensor, stacked_layers, all_hidden_batches

    if need_attn and all_attn_batches:
        num_layers = len(all_attn_batches[0])
        stacked_attns = []
        for l in range(num_layers):
            stacked_attns.append(torch.cat([b[l] for b in all_attn_batches], dim=0))
        attn_tensor = torch.stack(stacked_attns, dim=0)
        attn_path = cache_dir / "teacher_attn.pt"
        torch.save(attn_tensor, attn_path, _use_new_zipfile_serialization=True)
        sz = attn_path.stat().st_size
        manifest["attn_file"] = "teacher_attn.pt"
        manifest["attn_size_bytes"] = sz
        total_size_bytes += sz
        del attn_tensor, stacked_attns, all_attn_batches

    build_time = time.time() - t0
    manifest["build_time_s"] = round(build_time, 2)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_gb = total_size_bytes / 1e9
    LOG.info(
        "Cache built: %s (logits: %.1f GB, total: %.1f GB)",
        cache_dir, logits_gb, total_gb
    )
    return manifest_path


def load_teacher_cache(cache_dir: Path) -> Dict[str, Any]:
    """
    Loads cached outputs. Returns a dict:
      {
        "logits": Tensor(n_chunks, seq_len, vocab_size) fp16,
        "hidden_states": Tensor(n_layers, n_chunks, seq_len, hidden) fp16,
        "attentions": Tensor(n_layers, n_chunks, n_heads, seq_len, seq_len) fp16,
                        (or None if not cached),
        "manifest": {...}
      }
    """
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cache: Dict[str, Any] = {
        "manifest": manifest,
        "logits": None,
        "hidden_states": None,
        "attentions": None,
    }

    if manifest.get("has_logits") and manifest.get("logits_file"):
        logits_path = cache_dir / manifest["logits_file"]
        if logits_path.exists():
            try:
                cache["logits"] = torch.load(logits_path, map_location="cpu", mmap=True)
            except Exception:
                cache["logits"] = torch.load(logits_path, map_location="cpu")

    if manifest.get("has_hidden") and manifest.get("hidden_file"):
        hidden_path = cache_dir / manifest["hidden_file"]
        if hidden_path.exists():
            try:
                cache["hidden_states"] = torch.load(hidden_path, map_location="cpu", mmap=True)
            except Exception:
                cache["hidden_states"] = torch.load(hidden_path, map_location="cpu")

    if manifest.get("has_attn") and manifest.get("attn_file"):
        attn_path = cache_dir / manifest["attn_file"]
        if attn_path.exists():
            try:
                cache["attentions"] = torch.load(attn_path, map_location="cpu", mmap=True)
            except Exception:
                cache["attentions"] = torch.load(attn_path, map_location="cpu")

    return cache


def can_use_cache_for_method(
    cache: Optional[Dict[str, Any]],
    method_name: str,
    include_attn: bool = False,
) -> bool:
    if cache is None:
        return False
    if method_name == "A":
        return cache.get("logits") is not None
    elif method_name == "B":
        return cache.get("hidden_states") is not None
    elif method_name == "C":
        has_l = cache.get("logits") is not None
        has_h = cache.get("hidden_states") is not None
        has_a = (cache.get("attentions") is not None) if include_attn else True
        return has_l and has_h and has_a
    elif method_name == "D":
        return cache.get("logits") is not None and cache.get("hidden_states") is not None
    return False


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
    teacher_cache: Optional[Dict[str, Any]] = None,
    baseline_ppl: float = 21.71,
) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info(f"STARTING METHOD {method_name}")
    LOG.info("=" * 70)

    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_cache = can_use_cache_for_method(teacher_cache, method_name, getattr(args, "cache_attn", False))
    if use_cache:
        LOG.info("[%s] Using precomputed teacher cache (teacher forward pass BYPASS)", method_name)
    else:
        LOG.info("[%s] Teacher cache not active/available; running live teacher forward", method_name)

    # Instantiate fresh student from snapshot
    student = copy.deepcopy(base_snapshot)
    student.to(device)

    # Inject module
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

    # Configure train scope
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
        if not use_cache:
            base_teacher.eval()

        # Batch preparation with OOM fallback handling
        try:
            batch_slice = slice(chunk_idx, min(chunk_idx + args.batch_size, num_chunks))
            batch_ids = train_chunks[batch_slice]
            chunk_idx = (chunk_idx + args.batch_size) % num_chunks
            if batch_ids.size(1) > current_seq_len:
                batch_ids = batch_ids[:, :current_seq_len]
            batch_ids = batch_ids.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.startswith("cuda")):
                # Teacher forward or cached lookup
                if use_cache and teacher_cache is not None:
                    c_logits = None
                    if teacher_cache.get("logits") is not None:
                        c_logits = teacher_cache["logits"][batch_slice].to(device, non_blocking=True)
                        if c_logits.size(1) > current_seq_len:
                            c_logits = c_logits[:, :current_seq_len, :]

                    c_hidden = None
                    if teacher_cache.get("hidden_states") is not None:
                        raw_h = teacher_cache["hidden_states"]
                        c_hidden = tuple(
                            raw_h[l, batch_slice][:, :current_seq_len, :].to(device, non_blocking=True)
                            for l in range(raw_h.shape[0])
                        )

                    c_attn = None
                    if teacher_cache.get("attentions") is not None:
                        raw_a = teacher_cache["attentions"]
                        c_attn = tuple(
                            raw_a[l, batch_slice][:, :, :current_seq_len, :current_seq_len].to(device, non_blocking=True)
                            for l in range(raw_a.shape[0])
                        )

                    teacher_out = CachedTeacherOutput(
                        logits=c_logits,
                        hidden_states=c_hidden,
                        attentions=c_attn,
                    )
                else:
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
                    kl_temp = getattr(args, "kl_temperature", 2.0)
                    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=kl_temp)
                    shift_logits = student_out.logits[..., :-1, :].contiguous()
                    shift_labels = batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    kl_warmup = getattr(args, "kl_warmup_steps", 0)
                    if kl_warmup > 0:
                        kl_weight = 0.9 * min(1.0, float(step + 1) / float(max(1, kl_warmup)))
                    else:
                        kl_weight = 0.9
                    loss = kl_weight * loss_kl + 0.1 * loss_ce
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
                max_norm = getattr(args, "max_grad_norm", 1.0)
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % 50 == 0 or step == total_steps or step <= 5 or (total_steps <= 20 and step % 2 == 0):
                    step_time_ms = ((time.time() - t_start) / max(1, step)) * 1000.0
                    LOG.info(
                        f"[{method_name}] Step {step}/{total_steps} | Loss: {final_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Step time: {step_time_ms:.1f}ms"
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
    s_per_step = round(train_time / max(1, step), 4)
    peak_vram = (
        round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )

    # Final PPL
    post_ppl = ppl_history.get(step, float("nan"))
    if math.isnan(post_ppl) and ppl_history:
        post_ppl = list(ppl_history.values())[-1]

    ppl_ratio = round(post_ppl / baseline_ppl, 4) if np.isfinite(post_ppl) and post_ppl > 0 else float("inf")
    recovery_pct = (
        round(max(0.0, min(100.0, (pre_ppl - post_ppl) / (pre_ppl - baseline_ppl) * 100.0)), 2)
        if np.isfinite(post_ppl) and (pre_ppl - baseline_ppl) > 0
        else 0.0
    )

    LOG.info(
        f"[{method_name}] COMPLETE | Pre: {pre_ppl:.2f} -> Post: {post_ppl:.2f} | Recovery: {recovery_pct}% | VRAM: {peak_vram} MB | Time: {train_time}s ({s_per_step*1000:.1f}ms/step)"
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
        "s_per_step": s_per_step,
        "loss_final": round(final_loss, 4),
        "cached": bool(use_cache),
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
    header = "+---------+----------+----------+----------+----------+----------+----------+\n" \
             "| Method  | Pre PPL  | Post PPL | Recovery | Time (s) | VRAM(MB) | Cached   |\n" \
             "+---------+----------+----------+----------+----------+----------+----------+"
    print(header)
    for r in results:
        m = r["method"]
        pre = f"{r['pre_ppl']:.2f}" if isinstance(r["pre_ppl"], (int, float)) else str(r["pre_ppl"])
        post = f"{r['post_ppl']:.2f}" if isinstance(r["post_ppl"], (int, float)) else str(r["post_ppl"])
        rec = f"{r['recovery_pct']:.1f}%" if isinstance(r["recovery_pct"], (int, float)) else str(r["recovery_pct"])
        t = f"{r['train_time_s']:.1f}" if isinstance(r["train_time_s"], (int, float)) else str(r["train_time_s"])
        vram = f"{r['peak_vram_mb']:.1f}" if isinstance(r["peak_vram_mb"], (int, float)) else str(r["peak_vram_mb"])
        cached_str = "YES" if r.get("cached") else "NO"
        print(f"| {m:<7} | {pre:>8} | {post:>8} | {rec:>8} | {t:>8} | {vram:>8} | {cached_str:>8} |")
    print("+---------+----------+----------+----------+----------+----------+----------+")


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

    # Preprocessing Cache arguments
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store cache (default: cache/teacher_<hash>)")
    parser.add_argument("--cache_methods", nargs="+", default=None, help="Which methods need cache")
    parser.add_argument("--cache_force_rebuild", action="store_true", default=False, help="Rebuild cache even if it exists")
    parser.add_argument("--cache_attn", action="store_true", default=False, help="Also cache attention probs (large!)")
    parser.add_argument("--cache_max_train_texts", type=int, default=1000, help="Limit cache size (default 1000)")
    parser.add_argument("--no_cache", action="store_true", default=False, help="Disable teacher caching entirely")

    # Regularization & Distillation stability
    parser.add_argument("--kl_temperature", type=float, default=2.0, help="Temperature for KL divergence (default 2.0)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (default 1.0)")
    parser.add_argument("--kl_warmup_steps", type=int, default=0, help="Number of steps to linearly ramp up KL loss weight (default 0)")

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
    LOG.info("  Teacher Cache:    %s", "DISABLED" if args.no_cache else "ENABLED")
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
    max_train_texts = args.max_train_texts
    if args.cache_max_train_texts is not None and args.cache_max_train_texts < max_train_texts:
        max_train_texts = args.cache_max_train_texts

    if max_train_texts > 2000 and not args.no_cache:
        LOG.warning(
            "cache_max_train_texts (%d) > 2000: Cache file can be large (>30 GB). Consider --cache_max_train_texts 1000.",
            max_train_texts,
        )

    LOG.info("Preparing training chunks (max_texts=%d, seq_len=%d)...", max_train_texts, args.seq_len)
    train_chunks = build_train_chunks(ds_train, tokenizer, max_train_texts, args.seq_len)
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

    # 4.5 Teacher Output Preprocessing Cache
    teacher_cache = None
    if not args.no_cache:
        cache_methods = args.cache_methods or args.methods
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            c_hash = get_cache_hash(
                tok_src,
                tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "gpt2",
                args.seq_len,
                len(train_chunks),
                cache_methods,
                args.cache_attn,
            )
            cache_dir = Path("cache") / f"teacher_{c_hash}"

        need_build = args.cache_force_rebuild or not is_cache_valid(
            cache_dir,
            len(train_chunks),
            args.seq_len,
            cache_methods,
            args.cache_attn,
        )
        if need_build:
            build_teacher_cache(
                teacher=teacher,
                tokenizer=tokenizer,
                train_chunks=train_chunks,
                cache_dir=cache_dir,
                methods=cache_methods,
                device=args.device,
                seq_len=args.seq_len,
                include_attn=args.cache_attn,
                max_train_texts=max_train_texts,
            )
        else:
            LOG.info("Reusing existing teacher cache from %s", cache_dir)
        teacher_cache = load_teacher_cache(cache_dir)

    # 5. Run Each Method Sequentially
    results: List[Dict[str, Any]] = []
    for method in args.methods:
        try:
            res = run_kd_method(
                method, teacher, base_snapshot, tokenizer, train_chunks, ds_test, args,
                teacher_cache=teacher_cache, baseline_ppl=baseline_ppl,
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
        "peak_vram_mb", "train_time_s", "s_per_step", "loss_final", "cached",
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
