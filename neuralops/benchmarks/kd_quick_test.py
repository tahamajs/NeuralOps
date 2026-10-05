#!/usr/bin/env python3
"""
Adaptive Knowledge Distillation (KD) Experiment on MambaBlock.
Optimized for Single-Process, Multi-Module Continuous Sweep.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent if HERE.name in ("experiments", "neuralops") else HERE
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "neuralops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from neuralops.eval_all_models_datasets import apply_module_to_model

LOG = logging.getLogger("MambaKD")


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
def setup_logging(log_file: str | None = None):
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


def make_run_dir(module_name: str, args: argparse.Namespace) -> Path:
    timestamp = getattr(args, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    config_parts = [
        f"bs{args.batch_size}",
        f"seq{args.seq_len}",
        f"lr{args.learning_rate:.0e}",
    ]
    if getattr(args, "no_cosine_lr", False):
        config_parts.append("noclr")
    if args.methods:
        config_parts.append("".join(sorted(args.methods)))
    config_suffix = "_".join(config_parts)

    run_dir = Path("results") / "kd_runs" / module_name / f"{timestamp}_{config_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_registry(module_name: str, run_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace, baseline_ppl: float):
    try:
        registry_path = Path("results") / "kd_runs" / "registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"runs": []}
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text(encoding="utf-8"))
                if "runs" not in data or not isinstance(data["runs"], list):
                    data = {"runs": []}
            except Exception:
                data = {"runs": []}

        valid_results = [r for r in results if isinstance(r.get("post_ppl"), (int, float)) and np.isfinite(r["post_ppl"])]
        best_post_ppl = min((r["post_ppl"] for r in valid_results), default=float("nan"))
        best_recovery = max((r.get("recovery_pct", 0.0) for r in valid_results), default=0.0)

        entry = {
            "module": module_name,
            "timestamp": getattr(args, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")),
            "run_dir": str(run_dir).replace("\\", "/"),
            "methods": args.methods,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "seq_len": args.seq_len,
            "recovery_steps": args.recovery_steps,
            "baseline_ppl": baseline_ppl,
            "best_post_ppl": best_post_ppl,
            "best_recovery_pct": best_recovery,
            "results": {r["method"]: {k: v for k, v in r.items() if k != "method"} for r in results},
        }
        data["runs"].append(entry)
        registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        LOG.warning("Failed to update global registry: %s", e)


def update_latest_link(module_name: str, run_dir: Path):
    try:
        module_dir = Path("results") / "kd_runs" / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        latest_file = module_dir / "latest.txt"
        latest_file.write_text(str(run_dir.resolve()), encoding="utf-8")
    except Exception as e:
        LOG.warning("Failed to update latest.txt for %s: %s", module_name, e)


# -----------------------------------------------------------------------------
# Data Loading & Tokenization
# -----------------------------------------------------------------------------
def build_train_chunks(train_dataset, tokenizer, max_train_texts: int = 5000, seq_len: int = 256) -> torch.Tensor:
    texts: list[str] = []
    for item in train_dataset["text"]:
        t = item.strip()
        if len(t) > 20:
            texts.append(t)
        if len(texts) >= max_train_texts:
            break

    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog."] * 512

    all_ids: list[int] = []
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
def compute_perplexity(model: nn.Module, tokenizer, test_dataset, seq_len: int = 1024, stride: int = 512, max_eval_tokens: int = 15000, device: str = "cuda") -> float:
    model.eval()
    sample_texts = []
    accum_chars = 0
    target_chars = max_eval_tokens * 6
    for t in test_dataset["text"]:
        sample_texts.append(t)
        accum_chars += len(t)
        if accum_chars >= target_chars:
            break
    text = "\n\n".join(sample_texts) if sample_texts else "\n\n".join(test_dataset["text"][:100])
    enc = tokenizer(text, return_tensors="pt", truncation=False)
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

            with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=device.startswith("cuda")):
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
# KD Loss Functions & Adapters
# -----------------------------------------------------------------------------
class HiddenProjectionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 12):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_layers)])
        for p in self.projections:
            nn.init.eye_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, layer_idx: int, student_h: torch.Tensor) -> torch.Tensor:
        idx = min(layer_idx, len(self.projections) - 1)
        return self.projections[idx](student_h)


def compute_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    p_s = F.log_softmax(student_logits.view(-1, student_logits.size(-1)) / temperature, dim=-1)
    p_t = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (temperature * temperature)


def compute_hidden_mse(student_hidden_states: tuple[torch.Tensor, ...], teacher_hidden_states: tuple[torch.Tensor, ...], projection_adapter: HiddenProjectionAdapter | None = None) -> torch.Tensor:
    losses = []
    for idx, (s_h, t_h) in enumerate(zip(student_hidden_states[1:], teacher_hidden_states[1:])):
        s_aligned = projection_adapter(idx, s_h.float()) if projection_adapter is not None else s_h.float()
        losses.append(F.mse_loss(s_aligned, t_h.float()))
    return sum(losses) / max(1, len(losses))


def compute_attention_mse(student_attentions: tuple[torch.Tensor, ...] | None, teacher_attentions: tuple[torch.Tensor, ...] | None, device: str) -> torch.Tensor:
    if student_attentions is None or teacher_attentions is None:
        return torch.tensor(0.0, device=device)
    losses = []
    for s_a, t_a in zip(student_attentions, teacher_attentions):
        if s_a is not None and t_a is not None:
            losses.append(F.mse_loss(s_a.float(), t_a.float()))
    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


def compute_method_d_loss(student_out: Any, teacher_out: Any, input_ids: torch.Tensor, device: str, kl_temperature: float = 4.0, kl_weight: float = 0.5, hidden_weight: float = 0.4, ce_weight: float = 0.1, projection_adapter: HiddenProjectionAdapter | None = None) -> torch.Tensor:
    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=kl_temperature)
    cos_losses = []
    for idx, (s_h, t_h) in enumerate(zip(student_out.hidden_states[1:], teacher_out.hidden_states[1:])):
        s_aligned = projection_adapter(idx, s_h.float()) if projection_adapter is not None else s_h.float()
        cos_sim = F.cosine_similarity(s_aligned, t_h.float(), dim=-1)
        cos_losses.append((1.0 - cos_sim).mean())
    hidden_d = sum(cos_losses) / max(1, len(cos_losses))

    shift_logits = student_out.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return kl_weight * loss_kl + hidden_weight * hidden_d + ce_weight * loss_ce


def set_train_scope(model, train_scope="all"):
    if train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif train_scope in ("mamba", "attention", "module"):
        ATTN_ATTRS = ("attn", "attention", "self_attn", "mamba", "ssm")
        n_matched = 0
        for name, p in model.named_parameters():
            has_attn = any(attr in name.lower() for attr in ATTN_ATTRS)
            if has_attn:
                p.requires_grad = True
                n_matched += 1
            else:
                p.requires_grad = False
        LOG.info("set_train_scope: matched %d target param tensors", n_matched)
    else:
        raise ValueError(f"Unknown train_scope: {train_scope}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable

# -----------------------------------------------------------------------------
# Teacher Output Preprocessing Cache
# -----------------------------------------------------------------------------
class CachedTeacherOutput:
    def __init__(self, logits: torch.Tensor | None = None, hidden_states: tuple[torch.Tensor, ...] | None = None, attentions: tuple[torch.Tensor, ...] | None = None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


def get_cache_hash(model_path: str, tokenizer_name: str, seq_len: int, num_chunks: int, methods: list[str], include_attn: bool) -> str:
    cache_key_data = {
        "model_path": str(model_path), "tokenizer_name": str(tokenizer_name), "seq_len": int(seq_len), "num_chunks": int(num_chunks),
        "methods": sorted(list(set(methods))), "include_attn": bool(include_attn), "torch_version": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    return hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()[:12]


def is_cache_valid(cache_dir: Path, expected_chunks: int, expected_seq_len: int, methods: list[str], include_attn: bool = False) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("n_chunks") != expected_chunks or manifest.get("seq_len") != expected_seq_len:
            return False
        methods_set = set(methods)
        need_logits = any(m in methods_set for m in ("A", "C", "D"))
        need_hidden = any(m in methods_set for m in ("B", "C", "D"))
        need_attn = include_attn and ("C" in methods_set)

        if need_logits:
            fname = manifest.get("logits_file")
            if not manifest.get("has_logits") or not fname or not (cache_dir / fname).exists(): return False
        if need_hidden:
            fname = manifest.get("hidden_file")
            if not manifest.get("has_hidden") or not fname or not (cache_dir / fname).exists(): return False
        if need_attn:
            fname = manifest.get("attn_file")
            if not manifest.get("has_attn") or not fname or not (cache_dir / fname).exists(): return False
        return True
    except Exception:
        return False


def build_teacher_cache(teacher: nn.Module, tokenizer, train_chunks: torch.Tensor, cache_dir: Path, methods: list[str], device: str, seq_len: int, include_attn: bool = False, max_train_texts: int = 1000) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = len(train_chunks)
    methods_set = set(methods)
    need_logits = any(m in methods_set for m in ("A", "C", "D"))
    need_hidden = any(m in methods_set for m in ("B", "C", "D"))
    need_attn = include_attn and ("C" in methods_set)

    LOG.info("Building teacher cache for %d chunks...", n_chunks)
    t0 = time.time()
    teacher.eval()
    teacher.to(device)

    eval_batch_size = min(8, n_chunks)
    vocab_size = getattr(getattr(teacher, "config", None), "vocab_size", len(tokenizer))
    num_layers = getattr(getattr(teacher, "config", None), "n_layer", 12) + 1
    hidden_dim = getattr(getattr(teacher, "config", None), "n_embd", 768)

    logits_buffer = torch.empty((n_chunks, seq_len, vocab_size), dtype=torch.float16) if need_logits else None
    hidden_buffer = torch.empty((num_layers, n_chunks, seq_len, hidden_dim), dtype=torch.float16) if need_hidden else None
    attn_buffer = None
    if need_attn:
        n_heads = getattr(getattr(teacher, "config", None), "n_head", 12)
        n_attn_layers = getattr(getattr(teacher, "config", None), "n_layer", 12)
        attn_buffer = torch.empty((n_attn_layers, n_chunks, n_heads, seq_len, seq_len), dtype=torch.float16)

    with torch.no_grad():
        for start_idx in range(0, n_chunks, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_chunks)
            chunk_batch = train_chunks[start_idx:end_idx].to(device)
            if chunk_batch.size(1) > seq_len:
                chunk_batch = chunk_batch[:, :seq_len]

            with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=device.startswith("cuda")):
                out = teacher(chunk_batch, output_hidden_states=need_hidden, output_attentions=need_attn)

            if need_logits and logits_buffer is not None:
                logits_buffer[start_idx:end_idx] = out.logits.detach().to(torch.float16).cpu()
            if need_hidden and hidden_buffer is not None and out.hidden_states is not None:
                for l, h in enumerate(out.hidden_states):
                    hidden_buffer[l, start_idx:end_idx] = h.detach().to(torch.float16).cpu()
            if need_attn and attn_buffer is not None and out.attentions is not None:
                for l, a in enumerate(out.attentions):
                    attn_buffer[l, start_idx:end_idx] = a.detach().to(torch.float16).cpu()

    manifest: dict[str, Any] = {
        "n_chunks": n_chunks, "seq_len": seq_len, "vocab_size": vocab_size,
        "has_logits": need_logits, "has_hidden": need_hidden, "has_attn": need_attn,
        "logits_file": None, "hidden_file": None, "attn_file": None,
        "logits_size_bytes": 0, "hidden_size_bytes": 0, "attn_size_bytes": 0, "build_time_s": round(time.time() - t0, 2),
    }

    if need_logits and logits_buffer is not None:
        logits_path = cache_dir / "teacher_logits.pt"
        torch.save(logits_buffer, logits_path, _use_new_zipfile_serialization=True)
        manifest["logits_file"] = "teacher_logits.pt"
        del logits_buffer

    if need_hidden and hidden_buffer is not None:
        hidden_path = cache_dir / "teacher_hidden.pt"
        torch.save(hidden_buffer, hidden_path, _use_new_zipfile_serialization=True)
        manifest["hidden_file"] = "teacher_hidden.pt"
        del hidden_buffer

    if need_attn and attn_buffer is not None:
        attn_path = cache_dir / "teacher_attn.pt"
        torch.save(attn_buffer, attn_path, _use_new_zipfile_serialization=True)
        manifest["attn_file"] = "teacher_attn.pt"
        del attn_buffer

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Cache built: %s", cache_dir)
    return manifest_path


def load_teacher_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache: dict[str, Any] = {"manifest": manifest, "logits": None, "hidden_states": None, "attentions": None}

    def _load_tensor(key: str):
        if manifest.get(f"has_{key}") and manifest.get(f"{key}_file"):
            fpath = cache_dir / manifest[f"{key}_file"]
            if fpath.exists():
                try:
                    cache[key] = torch.load(fpath, map_location="cpu", weights_only=False)
                except Exception:
                    cache[key] = torch.load(fpath, map_location="cpu", mmap=True, weights_only=False)

    _load_tensor("logits")
    _load_tensor("hidden_states")
    _load_tensor("attentions")
    return cache


def can_use_cache_for_method(cache: dict[str, Any] | None, method_name: str, include_attn: bool = False) -> bool:
    if cache is None: return False
    if method_name == "A": return cache.get("logits") is not None
    elif method_name == "B": return cache.get("hidden_states") is not None
    elif method_name == "C": return (cache.get("logits") is not None and cache.get("hidden_states") is not None and (not include_attn or cache.get("attentions") is not None))
    elif method_name == "D": return cache.get("logits") is not None and cache.get("hidden_states") is not None
    return False


# -----------------------------------------------------------------------------
# Single Method Runner
# -----------------------------------------------------------------------------
def run_kd_method(
    method_name: str, module_name: str, base_teacher: AutoModelForCausalLM, base_snapshot: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, train_chunks: torch.Tensor, test_dataset: Any, args: argparse.Namespace,
    teacher_cache: dict[str, Any] | None = None, baseline_ppl: float = 21.71,
) -> dict[str, Any]:
    
    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_cache = can_use_cache_for_method(teacher_cache, method_name, getattr(args, "cache_attn", False))
    
    # Safe Graph Reset: Deepcopy from CPU guarantees a clean architectural state
    student = copy.deepcopy(base_snapshot)
    student.to(device)

    calib_slice = train_chunks[:4, :128].to(device)
    student, comp_stats = apply_module_to_model(student, module_name, "auto", rank=args.rank, device=device, calibration_data=calib_slice, tokenizer=tokenizer)

    if args.fp16 and device.startswith("cuda"):
        student.half()

    pre_ppl = compute_perplexity(student, tokenizer, test_dataset, seq_len=args.seq_len, stride=128, max_eval_tokens=args.eval_max_tokens, device=device)
    LOG.info("Pre-KD PPL for Module %s, Method %s: %.2f", module_name, method_name, pre_ppl)

    n_trainable = set_train_scope(student, args.train_scope)
    if n_trainable == 0:
        for name, p in student.named_parameters():
            if any(k in name.lower() for k in ["attn", "mamba", "ssm"]): 
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)

    for p in student.parameters():
        if p.requires_grad: p.data = p.data.float()

    projection_adapter = None
    if method_name in ("B", "C", "D"):
        projection_adapter = HiddenProjectionAdapter(hidden_dim=getattr(student.config, "n_embd", 768), num_layers=getattr(student.config, "n_layer", 12)).to(device)
        for p in projection_adapter.parameters(): p.data = p.data.float()

    attn_mamba_params, other_params = [], []
    for name, p in student.named_parameters():
        if p.requires_grad:
            if any(k in name.lower() for k in ["attn", "mamba", "ssm"]):
                attn_mamba_params.append(p)
            else: 
                other_params.append(p)

    adapter_params = list(projection_adapter.parameters()) if projection_adapter is not None else []
    param_groups = []
    if attn_mamba_params: param_groups.append({"params": attn_mamba_params, "lr": args.learning_rate * 5.0})
    if other_params: param_groups.append({"params": other_params, "lr": args.learning_rate})
    if adapter_params: param_groups.append({"params": adapter_params, "lr": args.learning_rate * 2.0})

    if not param_groups:
        for p in student.parameters():
            p.requires_grad = True
            p.data = p.data.float()
        param_groups = [{"params": [p for p in student.parameters() if p.requires_grad], "lr": args.learning_rate}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    all_trainable_params = [p for group in param_groups for p in group["params"]]

    total_steps = args.recovery_steps
    warmup_steps = args.warmup_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        if getattr(args, "no_cosine_lr", False): return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = bool(args.fp16 and device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    ppl_history: dict[int, float] = {}
    loss_history: list[tuple[int, float]] = []
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
        if not use_cache: base_teacher.eval()

        try:
            batch_slice = slice(chunk_idx, min(chunk_idx + args.batch_size, num_chunks))
            # VRAM optimization: Use non_blocking to overlap transfer with compute
            batch_ids = train_chunks[batch_slice].to(device, non_blocking=True)
            chunk_idx = (chunk_idx + args.batch_size) % num_chunks
            if batch_ids.size(1) > current_seq_len:
                batch_ids = batch_ids[:, :current_seq_len]

            with torch.cuda.amp.autocast(enabled=use_scaler):
                if use_cache and teacher_cache is not None:
                    c_logits, c_hidden, c_attn = None, None, None
                    if teacher_cache.get("logits") is not None:
                        c_logits = teacher_cache["logits"][batch_slice].to(device, non_blocking=True)[:, :current_seq_len, :]
                    if teacher_cache.get("hidden_states") is not None:
                        raw_h = teacher_cache["hidden_states"]
                        c_hidden = tuple(raw_h[l, batch_slice][:, :current_seq_len, :].to(device, non_blocking=True) for l in range(raw_h.shape[0]))
                    if teacher_cache.get("attentions") is not None:
                        raw_a = teacher_cache["attentions"]
                        c_attn = tuple(raw_a[l, batch_slice][:, :, :current_seq_len, :current_seq_len].to(device, non_blocking=True) for l in range(raw_a.shape[0]))
                    teacher_out = CachedTeacherOutput(logits=c_logits, hidden_states=c_hidden, attentions=c_attn)
                else:
                    with torch.no_grad():
                        teacher_out = base_teacher(batch_ids, output_hidden_states=(method_name in ("B", "C", "D")), output_attentions=(method_name == "C"))

                student_out = student(batch_ids, output_hidden_states=(method_name in ("B", "C", "D")), output_attentions=(method_name == "C"))

                progress_ratio = float(step) / float(max(1, total_steps))
                phase_hidden_weight = max(0.2, 0.7 * (1.0 - progress_ratio))
                phase_kl_weight = min(0.7, 0.2 + 0.5 * progress_ratio)
                kl_warmup = getattr(args, "kl_warmup_steps", 0)
                warmup_factor = min(1.0, float(step + 1) / float(max(1, kl_warmup))) if kl_warmup > 0 else 1.0
                kl_temp = getattr(args, "kl_temperature", 2.0)

                if method_name == "A":
                    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=kl_temp)
                    shift_logits, shift_labels = student_out.logits[..., :-1, :].contiguous(), batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = (0.9 * warmup_factor) * loss_kl + 0.1 * loss_ce
                elif method_name == "B":
                    loss_hidden = compute_hidden_mse(student_out.hidden_states, teacher_out.hidden_states, projection_adapter)
                    shift_logits, shift_labels = student_out.logits[..., :-1, :].contiguous(), batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = 0.9 * loss_hidden + 0.1 * loss_ce
                elif method_name == "C":
                    loss_kl = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=kl_temp)
                    loss_hidden = compute_hidden_mse(student_out.hidden_states, teacher_out.hidden_states, projection_adapter)
                    loss_attn = compute_attention_mse(student_out.attentions, teacher_out.attentions, device)
                    shift_logits, shift_labels = student_out.logits[..., :-1, :].contiguous(), batch_ids[..., 1:].contiguous()
                    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = (phase_kl_weight * warmup_factor) * loss_kl + phase_hidden_weight * loss_hidden + 0.05 * loss_attn + 0.05 * loss_ce
                elif method_name == "D":
                    loss = compute_method_d_loss(student_out, teacher_out, batch_ids, device, kl_temperature=kl_temp, kl_weight=phase_kl_weight * warmup_factor, hidden_weight=phase_hidden_weight, ce_weight=0.1, projection_adapter=projection_adapter)

                loss_scaled = loss / args.grad_accum_steps
                final_loss = float(loss.item())

            if use_scaler: scaler.scale(loss_scaled).backward()
            else: loss_scaled.backward()
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                loss_history.append((step, final_loss))
                max_norm = getattr(args, "max_grad_norm", 1.0)
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=max_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % 50 == 0 or step == total_steps or step <= 5:
                    LOG.info(f"[{method_name}] Step {step}/{total_steps} | Loss: {final_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                if step % args.eval_every == 0 or step == total_steps:
                    eval_ppl = compute_perplexity(student, tokenizer, test_dataset, seq_len=current_seq_len, stride=128, max_eval_tokens=args.eval_max_tokens, device=device)
                    ppl_history[step] = eval_ppl
                    LOG.info(f"[{method_name}] Step {step} Eval PPL: {eval_ppl:.2f}")
                    if math.isnan(eval_ppl) or eval_ppl > args.early_stop_ppl:
                        LOG.warning(f"[{method_name}] Early stop triggered at step {step}: PPL={eval_ppl}")
                        break

        except torch.cuda.OutOfMemoryError:
            LOG.warning(f"[{method_name}] CUDA OOM! Halving sequence length to 128...")
            current_seq_len = 128
            optimizer.zero_grad()
            accum_count = 0  
            gc.collect()
            torch.cuda.empty_cache()
            continue

    train_time = round(time.time() - t_start, 2)
    s_per_step = round(train_time / max(1, step), 4)
    peak_vram = round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2) if device.startswith("cuda") and torch.cuda.is_available() else 0.0

    post_ppl = ppl_history.get(step, float("nan"))
    if math.isnan(post_ppl) and ppl_history: post_ppl = list(ppl_history.values())[-1]
    ppl_ratio = round(post_ppl / baseline_ppl, 4) if np.isfinite(post_ppl) and post_ppl > 0 else float("inf")
    recovery_pct = round(min(100.0, (pre_ppl - post_ppl) / (pre_ppl - baseline_ppl) * 100.0), 2) if np.isfinite(post_ppl) and (pre_ppl - baseline_ppl) > 0 else 0.0

    LOG.info(f"[{method_name}] COMPLETE | Pre: {pre_ppl:.2f} -> Post: {post_ppl:.2f} | Recovery: {recovery_pct}% | VRAM: {peak_vram} MB | Time: {train_time}s")

    # Thorough VRAM cleanup before next module/method
    del student
    if projection_adapter is not None: del projection_adapter
    del optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    time.sleep(1)

    total_teacher_params = sum(p.numel() for p in base_teacher.parameters())
    total_student_params = sum(p.numel() for p in student.parameters())

    return {
        "method": method_name, "pre_ppl": pre_ppl, "post_ppl": post_ppl, "ppl_ratio": ppl_ratio,
        "recovery_pct": recovery_pct, "peak_vram_mb": peak_vram, "train_time_s": train_time,
        "s_per_step": s_per_step, "loss_final": round(final_loss, 4), "cached": bool(use_cache),
        "teacher_params": total_teacher_params,
        "student_params": total_student_params,
        "trainable_params": n_trainable,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "recovery_steps": args.recovery_steps,
        "train_scope": args.train_scope,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

# -----------------------------------------------------------------------------
# Comparison Table & Main
# -----------------------------------------------------------------------------
def print_comparison_table(results: list[dict[str, Any]]):
    header = "+---------+----------+----------+----------+----------+----------+----------+\n| Method  | Pre PPL  | Post PPL | Recovery | Time (s) | VRAM(MB) | Cached   |\n+---------+----------+----------+----------+----------+----------+----------+"
    print(header)
    for r in results:
        m = r["method"]
        pre, post, rec = f"{r['pre_ppl']:.2f}", f"{r['post_ppl']:.2f}", f"{r['recovery_pct']:.1f}%"
        t, vram, cached = f"{r['train_time_s']:.1f}", f"{r['peak_vram_mb']:.1f}", "YES" if r.get("cached") else "NO"
        print(f"| {m:<7} | {pre:>8} | {post:>8} | {rec:>8} | {t:>8} | {vram:>8} | {cached:>8} |")
    print("+---------+----------+----------+----------+----------+----------+----------+")

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation on injected Mamba module")
    
    # ---- Model & Data ----
    parser.add_argument("--model_name", type=str, default="gpt2", help="Teacher/student base model")
    # FIX: Native Multi-Module support
    parser.add_argument("--modules", nargs="+", default=["mamba"], help="List of modules to process sequentially")
    parser.add_argument("--rank", type=int, default=64, help="Rank for injected module")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max_train_texts", type=int, default=5000, help="Max texts for training chunks")
    parser.add_argument("--num_chunks", type=int, default=412, help="Max number of training chunks (0=all)")
    
    # ---- Training ----
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--recovery_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_max_tokens", type=int, default=15000)
    parser.add_argument("--early_stop_ppl", type=float, default=5000.0)
    parser.add_argument("--no_cosine_lr", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # ---- KD Config ----
    parser.add_argument("--methods", nargs="+", default=["C"], choices=["A", "B", "C", "D"],
                        help="KD methods: A=logit KL, B=hidden MSE, C=curriculum, D=cosine+logit")
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--kl_warmup_steps", type=int, default=100)
    
    # ---- Cache ----
    parser.add_argument("--use_cache", action="store_true", help="Use cached teacher outputs")
    parser.add_argument("--cache_dir", type=str, default=None, help="Custom cache directory")
    parser.add_argument("--cache_attn", action="store_true", help="Cache attention maps too")
    
    # ---- System ----
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", default=(torch.cuda.is_available()))
    parser.add_argument("--train_scope", type=str, default="module",
                        choices=["all", "mamba", "attention", "module"])
    
    parser.add_argument("--baseline_ppl", type=float, default=None, help="Teacher baseline PPL (optional override)")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or f"logs/kd_sweep_{args.timestamp}.log"
    setup_logging(log_file)
    
    LOG.info("=" * 70)
    LOG.info("Multi-Module Knowledge Distillation Sweep")
    LOG.info("=" * 70)
    LOG.info("Model: %s | Modules: %d to process | Methods: %s", args.model_name, len(args.modules), args.methods)
    LOG.info("Device: %s | FP16: %s | Batch: %d | Seq: %d", args.device, args.fp16, args.batch_size, args.seq_len)
    
    # ---- 1. Phase 0: Setup Once ----
    LOG.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    LOG.info("Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    train_chunks = build_train_chunks(dataset["train"], tokenizer, max_train_texts=args.max_train_texts, seq_len=args.seq_len)
    if args.num_chunks > 0:
        train_chunks = train_chunks[:args.num_chunks]
    LOG.info("Training chunks: %s", tuple(train_chunks.shape))
    
    LOG.info("Loading teacher model: %s", args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name)
    teacher.eval()
    teacher.to(args.device)
    
    # Safe Graph Setup: Kept on CPU to avoid mutation and save VRAM
    LOG.info("Creating safe pristine student snapshot (CPU)...")
    base_snapshot = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_snapshot.eval()
    base_snapshot.cpu() 
    
    # ---- 2. Cache Generation & Baseline ----
    teacher_cache = None
    if args.use_cache:
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            cache_hash = get_cache_hash(args.model_name, args.model_name, args.seq_len, len(train_chunks), args.methods, args.cache_attn)
            cache_dir = Path("results") / "kd_cache" / f"{args.model_name}_{cache_hash}"
        
        if is_cache_valid(cache_dir, len(train_chunks), args.seq_len, args.methods, args.cache_attn):
            LOG.info("Loading existing teacher cache from %s", cache_dir)
            teacher_cache = load_teacher_cache(cache_dir)
        else:
            LOG.info("Building teacher cache at %s", cache_dir)
            build_teacher_cache(teacher, tokenizer, train_chunks, cache_dir, args.methods, args.device, args.seq_len, args.cache_attn, args.max_train_texts)
            teacher_cache = load_teacher_cache(cache_dir)
    
    LOG.info("Computing teacher baseline PPL...")
    baseline_ppl = compute_perplexity(teacher, tokenizer, dataset["test"], seq_len=args.seq_len, stride=128, max_eval_tokens=args.eval_max_tokens, device=args.device)
    LOG.info("Using measured teacher PPL as baseline: %.2f", baseline_ppl)

    # VRAM Optimization: Move Teacher to CPU if cache is actively handling everything
    if args.use_cache and teacher_cache is not None:
        LOG.info("Teacher cache is loaded. Evicting Teacher from VRAM to CPU to save memory...")
        teacher.cpu()
        torch.cuda.empty_cache()
    
    # ---- 3. Phase 1: Module Sweep Loop ----
    for idx, module_name in enumerate(args.modules, 1):
        LOG.info("\n" + "=" * 70)
        LOG.info(f"PROCESSING MODULE {idx}/{len(args.modules)}: {module_name}")
        LOG.info("=" * 70)
        
        run_dir = make_run_dir(module_name, args)
        module_results: list[dict[str, Any]] = []
        
        for method_name in args.methods:
            LOG.info("\n# Running method %s for %s", method_name, module_name)
            result = run_kd_method(
                method_name=method_name,
                module_name=module_name,
                base_teacher=teacher,
                base_snapshot=base_snapshot,
                tokenizer=tokenizer,
                train_chunks=train_chunks,
                test_dataset=dataset["test"],
                args=args,
                teacher_cache=teacher_cache,
                baseline_ppl=baseline_ppl,
            )
            module_results.append(result)
        
        print_comparison_table(module_results)
        
        # Incremental Save: Ensure data is saved immediately after EACH module
        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(module_results, indent=2), encoding="utf-8")
        
        update_registry(module_name, run_dir, module_results, args, baseline_ppl)
        update_latest_link(module_name, run_dir)
        
        if HAS_MATPLOTLIB and len(module_results) > 0:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                methods = [r["method"] for r in module_results]
                pre_ppls = [r["pre_ppl"] for r in module_results]
                post_ppls = [r["post_ppl"] for r in module_results]
                x = np.arange(len(methods))
                width = 0.35
                ax.bar(x - width/2, pre_ppls, width, label="Pre-KD PPL")
                ax.bar(x + width/2, post_ppls, width, label="Post-KD PPL")
                ax.axhline(y=baseline_ppl, color="r", linestyle="--", label=f"Teacher baseline ({baseline_ppl:.2f})")
                ax.set_xticks(x)
                ax.set_xticklabels(methods)
                ax.set_ylabel("Perplexity")
                ax.set_title(f"KD Recovery — {module_name}")
                ax.legend()
                ax.set_yscale("log")
                plt.tight_layout()
                plt.savefig(run_dir / "comparison.png", dpi=100)
                plt.close()
            except Exception as e:
                LOG.warning("Plot generation failed: %s", e)

    LOG.info("\n" + "=" * 70)
    LOG.info("SWEEP COMPLETELY FINISHED!")
    LOG.info("=" * 70)
    return 0

if __name__ == "__main__":
    main()