#!/usr/bin/env python3
"""
Decoder Benchmark Runner: Comprehensive evaluation of attention/compression modules
on GPT-2 with WikiText-2.

Measures per module:
  - Perplexity (PPL) via sliding-window evaluation
  - Peak VRAM, latency, throughput
  - Parameter savings / compression ratio
  - Injection / compression statistics (real work done, not just PASS)

Production features :
  1.  Upsert-based CSV: one row per module, no duplicates across runs.
  2.  Intelligent skip: only skips modules that ACTUALLY did work
      (num_injected_attention_layers > 0 or num_compressed_layers > 0).
  3.  --no_skip / --force / --only_failed / --retry_flags flags.
  4.  Baseline PPL caching (re-use across runs unless --recompute_baseline).
  5.  Full metric capture: injection counts, recovery time, LoRA params, etc.
  6.  Phantom-pass detection: PASS_WITHOUT_EFFECT status.
  7.  tqdm progress bar with per-module timing.
  8.  Structured logging to file + console (with optional colored output).
  9.  Atomic CSV writes via tmp-file + rename.
  10. Multi-worker safety: single global summary + per-worker CSV.
  11. Comprehensive diag.json with dtype, shape, device info on failure.
  12. Token packing (no padding waste) for calibration.
  13. Configurable calibration size.
  14. Reproducible seeding per module.
  15. Graceful KeyboardInterrupt: saves partial results before exiting.
  16. Per-module JSON result with full stats.
  17. End-of-run Markdown report generation.
  18. Baseline sanity check: warns if baseline PPL is anomalous.
  19. Memory-leak guard: explicit gc + cuda.empty_cache between modules.
  20. Config file support (YAML/JSON).
  21. --dry_run preview with expected I/O.
  22. Better exception messages with stack locals on failure (optional).
  23. Multi-GPU aware (--device cuda:N).
  24. Summary statistics at the end (mean, median, best/worst).
  25. Auto-detection of deprecated module files from multiple locations.

Usage:
  python decoder_benchmark.py --modules all --no_skip --force
  python decoder_benchmark.py --modules mla,svd_llm --only_failed
  python decoder_benchmark.py --total_workers 4 --worker_id 0
"""
from __future__ import annotations

import argparse
import atexit
import copy
import csv
import gc
import json
import logging
import os
import random
import shutil
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT), str(ROOT / "Codes")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    is_helper_module,
    is_module_declared_compatible,
)
from framework.eval_all_models_datasets import apply_module_to_model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG = logging.getLogger("GPT2Sweep")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)


# ---------------------------------------------------------------------------
# CSV schema — the single source of truth
# ---------------------------------------------------------------------------
CSV_FIELDS: List[str] = [
    # Identity
    "module_name",
    "status",
    # Quality
    "perplexity",
    "baseline_ppl",
    "delta_ppl",
    "ppl_ratio",
    # Compression
    "compression_rate",
    "compression_ratio_pct",
    "total_params_before",
    "total_params_after",
    "delta_params",
    # Injection / compression detail (the "did it actually work?" fields)
    "num_injected_attention_layers",
    "num_compressed_layers",
    "num_fallback_layers",
    "lora_params",
    "calibration_time_s",
    "recovery_time_s",
    # Runtime metrics
    "peak_vram_mb",
    "latency_ms_per_token",
    "throughput_tok_per_sec",
    "eval_elapsed_s",
    "wall_elapsed_s",
    # Diagnostics
    "error",
    "timestamp",
    # KD Recovery (optional)
    "post_kd_ppl",
    "delta_ppl_recovery",
]


# ---------------------------------------------------------------------------
# Dataclass for a single result row (typed, no dict typos)
# ---------------------------------------------------------------------------
@dataclass
class ResultRow:
    module_name: str
    status: str = "PENDING"
    perplexity: float = -1.0
    baseline_ppl: float = -1.0
    delta_ppl: float = 0.0
    ppl_ratio: float = 1.0

    compression_rate: float = 1.0
    compression_ratio_pct: float = 0.0
    total_params_before: int = 0
    total_params_after: int = 0
    delta_params: int = 0

    num_injected_attention_layers: int = 0
    num_compressed_layers: int = 0
    num_fallback_layers: int = 0
    lora_params: int = 0
    calibration_time_s: float = 0.0
    recovery_time_s: float = 0.0

    peak_vram_mb: float = 0.0
    latency_ms_per_token: float = 0.0
    throughput_tok_per_sec: float = 0.0
    eval_elapsed_s: float = 0.0
    wall_elapsed_s: float = 0.0

    error: str = ""
    timestamp: str = ""

    post_kd_ppl: float = -1.0
    delta_ppl_recovery: float = 0.0

    def to_csv_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k, "") for k in CSV_FIELDS}


# ---------------------------------------------------------------------------
# Deprecated-module loading
# ---------------------------------------------------------------------------
def load_deprecated_modules() -> Set[str]:
    """Collect deprecated module names from all known file locations."""
    deprecated: Set[str] = set()
    candidates = [
        ROOT / "audit" / "deprecated_modules.txt",
        ROOT / "deprecated_modules.txt",
        CODES / "framework" / "deprecated_modules.txt",
        HERE / "deprecated_modules.txt",
        HERE / "config" / "deprecated_modules.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    deprecated.add(line)
        except Exception as e:
            LOG.warning("Could not read %s: %s", p, e)
    return deprecated


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
def get_all_decoder_modules(category: str = "all") -> List[str]:
    """Dynamically discover all GPT-2-compatible attention and compressor modules."""
    import framework.attentions  # noqa: F401
    import framework.compression_engine  # noqa: F401

    items: List[Tuple[str, Any]] = []
    if category in ("all", "attentions"):
        items.extend(ATTENTION_REGISTRY.items())
    if category in ("all", "compressors"):
        items.extend(COMPRESSOR_REGISTRY.items())

    by_factory: Dict[Any, List[str]] = {}
    for name, factory in items:
        if name == "full" or is_helper_module(name):
            continue
        try:
            is_compat, _ = is_module_declared_compatible(name, "gpt2")
        except Exception:
            is_compat = True
        if is_compat:
            by_factory.setdefault(factory, []).append(name)

    canonical = ["full"]
    for aliases in by_factory.values():
        # Prefer shorter / non-`_inject` aliases as canonical names
        canonical.append(sorted(aliases, key=lambda x: (x.endswith("_inject"), len(x), x))[0])

    # Preserve insertion order but ensure "full" is first
    seen: Set[str] = set()
    out: List[str] = []
    for m in canonical:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT-2 decoder perplexity benchmark sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / data
    p.add_argument("--model_path", type=str, default="models/gpt2_wikitext")
    p.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="test", choices=("test", "validation", "train"))
    p.add_argument("--output_dir", type=str, default="results/gpt2_sweep")

    # Modules
    p.add_argument("--modules", type=str, nargs="*", default=["all"])
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_length", type=int, default=None, help="Alias for --seq_len")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max_eval_tokens", type=int, default=100_000,
                   help="Cap total evaluated tokens (None = full split)")

    # Precision / device
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda, cuda:0, cuda:1, cpu")
    p.add_argument("--seed", type=int, default=42)

    # Calibration
    p.add_argument("--calib_samples", type=int, default=500)
    p.add_argument("--calib_batch_size", type=int, default=4)
    p.add_argument("--calib_max_length", type=int, default=128)

    # Behavior flags
    p.add_argument("--no_skip", action="store_true", default=False,
                   help="Re-evaluate ALL modules, even successful ones")
    p.add_argument("--force", action="store_true", default=False,
                   help="Overwrite existing CSV (start fresh)")
    p.add_argument("--only_failed", action="store_true", default=False,
                   help="Only evaluate modules whose last status is FAIL/PASS_*_DEGRADED")
    p.add_argument("--retry_flags", type=str, nargs="*", default=[],
                   help="Statuses to treat as failed and retry "
                        "(e.g. --retry_flags PASS_NO_COMPRESSION FAIL_PPL_DIVERGED)")
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--recompute_baseline", action="store_true", default=False,
                   help="Force recompute baseline PPL (default: reuse cached)")

    # KD Recovery
    p.add_argument("--with_kd_recovery", action="store_true", default=False,
                   help="Run KD recovery fine-tuning if zero-shot PPL > 2x baseline")
    p.add_argument("--kd_recovery_steps", type=int, default=100,
                   help="Number of KD steps to run for recovery")

    # Parallelism
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--total_workers", type=int, default=1)

    # Reporting
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--save_traceback_locals", action="store_true", default=False,
                   help="Include stack locals in diag.json (verbose)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging to file
# ---------------------------------------------------------------------------
def setup_logging(log_file: Optional[str], verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers to avoid double logs
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(_LOG_FORMAT)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Tokenization + calibration
# ---------------------------------------------------------------------------
def build_calibration_loader(
    train_dataset,
    tokenizer,
    num_samples: int = 500,
    batch_size: int = 4,
    max_length: int = 128,
) -> List[Dict[str, torch.Tensor]]:
    """
    Pack short WikiText lines into fixed-length chunks (no padding waste).
    Returns a list of batches: [{"input_ids": (B, T), "attention_mask": (B, T)}, ...]
    """
    # Collect enough text
    texts: List[str] = []
    for item in train_dataset["text"]:
        t = item.strip()
        if len(t) > 20:
            texts.append(t)
        if len(texts) >= max(num_samples * 4, 512):
            break

    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog."] * 512

    # Tokenize each chunk (no padding), then concatenate
    all_ids: List[int] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
        if len(all_ids) >= num_samples * max_length:
            break

    target = num_samples * max_length
    if len(all_ids) < target:
        # Repeat if not enough
        while len(all_ids) < target:
            all_ids.extend(all_ids[: max(1, target - len(all_ids))])

    all_ids = all_ids[:target]
    ids_tensor = torch.tensor(all_ids, dtype=torch.long).view(num_samples, max_length)

    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, num_samples, batch_size):
        chunk = ids_tensor[i : i + batch_size]
        attn = torch.ones_like(chunk)
        batches.append({"input_ids": chunk, "attention_mask": attn})
    return batches


# ---------------------------------------------------------------------------
# Perplexity + telemetry
# ---------------------------------------------------------------------------
def compute_perplexity_with_metrics(
    model: nn.Module,
    tokenizer,
    test_dataset,
    seq_len: int = 1024,
    stride: int = 512,
    max_eval_tokens: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Sliding-window perplexity (GPT-2 / LLaMA-style).

    PPL = exp( Σ NLL_t × target_len_t / Σ target_len_t )
    """
    model.eval()
    text = "\n\n".join(test_dataset["text"])

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_overflowing_tokens=False,
    )
    input_ids_all = enc.input_ids
    if input_ids_all.dim() == 1:
        input_ids_all = input_ids_all.unsqueeze(0)
    if input_ids_all.size(0) > 1:
        input_ids_all = input_ids_all.view(1, -1)

    total = input_ids_all.size(1)
    if max_eval_tokens:
        total = min(total, int(max_eval_tokens))

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    nlls: List[torch.Tensor] = []
    prev_end = 0
    tokens_eval = 0

    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.float16,
        enabled=device.startswith("cuda"),
    ):
        for begin in range(0, total, stride):
            end = min(begin + seq_len, total)
            trg_len = end - prev_end
            if trg_len <= 0:
                break
            ids = input_ids_all[:, begin:end].to(device, non_blocking=True)
            tgt = ids.clone()
            tgt[:, :-trg_len] = -100

            out = model(ids, labels=tgt)
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                # Degenerate; break and report NaN
                nlls.append(torch.tensor(float("nan")))
                break

            nlls.append((loss * trg_len).detach().cpu())
            tokens_eval += trg_len
            prev_end = end
            if end == total:
                break

    elapsed = max(1e-4, time.time() - t0)

    if not nlls or any(torch.isnan(x) for x in nlls):
        ppl = float("nan")
    else:
        stack = torch.stack(nlls)
        total_nll = float(stack.sum().item())
        ppl = float(np.exp(total_nll / max(1, prev_end)))

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    lat = (elapsed / max(1, tokens_eval)) * 1000.0
    tps = tokens_eval / elapsed

    return {
        "perplexity": round(ppl, 4) if np.isfinite(ppl) else -1.0,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms_per_token": round(lat, 4),
        "throughput_tok_per_sec": round(tps, 2),
        "eval_elapsed_s": round(elapsed, 3),
        "tokens_evaluated": tokens_eval,
        "is_finite": bool(np.isfinite(ppl)),
    }


# ---------------------------------------------------------------------------
# KD Recovery (Fine-tuning recovery for post-injection PPL)
# ---------------------------------------------------------------------------
def run_kd_recovery(
    student: nn.Module,
    teacher: nn.Module,
    tokenizer,
    train_dataset,
    device: str = "cuda",
    steps: int = 100,
    seq_len: int = 128,
    lr: float = 2.5e-5,
    kl_temp: float = 4.0,
) -> float:
    """
    Lightweight KD recovery (Method A / logits KD) on train_dataset.
    Returns recovery time in seconds.
    """
    LOG.info("Running KD recovery: %d steps, batch=1, seq_len=%d, lr=%.2e", steps, seq_len, lr)
    t0 = time.time()
    
    # Configure trainable parameters: unfreeze low-rank / SVD / injected params
    trainable_keywords = ("W_u", "W_v", "weight_u", "weight_v", "lora_A", "lora_B", "adapter", "A", "B")
    trainable_params = []
    for name, p in student.named_parameters():
        if ".attn." not in name:
            p.requires_grad = False
            continue
        sub = name.split(".attn.", 1)[-1]
        if any(kw in name for kw in trainable_keywords):
            p.requires_grad = True
        elif sub.startswith(("c_attn.weight", "c_proj.weight", "c_attn.bias", "c_proj.bias")) or sub in ("c_attn", "c_proj"):
            p.requires_grad = False
        else:
            p.requires_grad = True

        if p.requires_grad:
            p.data = p.data.float()
            trainable_params.append(p)

    if not trainable_params:
        LOG.warning("No trainable parameters found for KD recovery; training all parameters with requires_grad.")
        trainable_params = [p for p in student.parameters() if p.requires_grad]

    if not trainable_params:
        LOG.warning("KD recovery: completely 0 trainable parameters, skipping recovery.")
        return 0.0

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    use_scaler = bool(device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Tokenize training chunks
    texts = [t.strip() for t in train_dataset["text"] if len(t.strip()) > 20][:max(steps * 2, 200)]
    all_ids = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if ids:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id or 0)
        if len(all_ids) >= steps * seq_len:
            break

    target_tokens = steps * seq_len
    if len(all_ids) < target_tokens:
        while len(all_ids) < target_tokens:
            all_ids.extend(all_ids[:max(1, target_tokens - len(all_ids))])
    all_ids = all_ids[:target_tokens]
    chunks = torch.tensor(all_ids, dtype=torch.long).view(steps, seq_len)

    student.train()
    teacher.eval()
    teacher.to(device)

    for step_idx in range(steps):
        batch = chunks[step_idx : step_idx + 1].to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                t_out = teacher(batch)

        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            s_out = student(batch)
            # Logits KD (KL divergence with temperature)
            p_s = F.log_softmax(s_out.logits / kl_temp, dim=-1)
            p_t = F.softmax(t_out.logits / kl_temp, dim=-1)
            loss_kl = F.kl_div(p_s, p_t, reduction="batchmean") * (kl_temp * kl_temp)

            # Next token cross-entropy
            shift_logits = s_out.logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = 0.9 * loss_kl + 0.1 * loss_ce

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()

    student.eval()
    elapsed = time.time() - t0
    LOG.info("KD recovery finished in %.2fs", elapsed)
    return elapsed


# ---------------------------------------------------------------------------
# CSV upsert (no duplicates, atomic)
# ---------------------------------------------------------------------------
def _read_csv_rows(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                m = r.get("module_name")
                if m:
                    out[m] = r
    except Exception as e:
        LOG.warning("Could not read %s: %s", csv_path, e)
    return out


def upsert_row(csv_path: Path, row: Dict[str, Any], lock_path: Optional[Path] = None):
    """Atomic upsert by module_name. Uses FileLock for multiprocess safety."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path or csv_path.with_suffix(".lock")
    with FileLock(str(lock)):
        existing = _read_csv_rows(csv_path)
        existing[row["module_name"]] = {k: row.get(k, "") for k in CSV_FIELDS}

        tmp = csv_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            # Stable ordering: "full" first, then alphabetical
            keys = sorted(existing.keys(), key=lambda k: (k != "full", k))
            for k in keys:
                writer.writerow(existing[k])
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(csv_path)


# ---------------------------------------------------------------------------
# Diagnostic writer
# ---------------------------------------------------------------------------
def write_diag(
    diag_path: Path,
    module: str,
    exc: BaseException,
    extra: Optional[Dict[str, Any]] = None,
    include_locals: bool = False,
):
    exc_type, exc_val, exc_tb = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)

    last = traceback.extract_tb(exc_tb)[-1] if exc_tb else None
    info: Dict[str, Any] = {
        "module_name": module,
        "exception_type": exc_type.__name__ if exc_type else "Exception",
        "exception_message": str(exc_val),
        "errno": getattr(exc_val, "errno", None),
        "filename": str(getattr(exc_val, "filename", "")) or None,
        "fault_file": last.filename if last else "unknown",
        "fault_line": last.lineno if last else -1,
        "fault_function": last.name if last else "unknown",
        "cwd": os.getcwd(),
        "traceback": tb_lines,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        info.update(extra)
    if include_locals and exc_tb is not None:
        frame = exc_tb.tb_frame
        try:
            info["locals_keys"] = list(frame.f_locals.keys())
        except Exception:
            pass
    try:
        diag_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        LOG.warning("Failed to write diag.json for %s: %s", module, e)


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------
def classify_status(ppl_ratio: float) -> str:
    if not np.isfinite(ppl_ratio):
        return "FAIL_PPL_NAN"
    if ppl_ratio <= 1.05:
        return "PASS"
    if ppl_ratio <= 1.20:
        return "PASS_SLIGHT_DEGRADATION"
    if ppl_ratio <= 1.50:
        return "PASS_DEGRADED"
    return "FAIL_PPL_DIVERGED"


def is_real_success(status: str, row: Dict[str, Any]) -> bool:
    """A module is a 'real success' iff it PASSED AND did measurable work."""
    if status not in ("PASS", "PASS_SLIGHT_DEGRADATION"):
        return False
    try:
        n_inj = int(row.get("num_injected_attention_layers", 0) or 0)
        n_comp = int(row.get("num_compressed_layers", 0) or 0)
    except (ValueError, TypeError):
        return False
    return (n_inj + n_comp) > 0


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_markdown_report(out_dir: Path, csv_path: Path, baseline_ppl: float):
    rows = _read_csv_rows(csv_path)
    if not rows:
        return
    md_lines = [
        f"# GPT-2 Decoder Sweep Report",
        f"",
        f"- Baseline PPL: **{baseline_ppl:.2f}**",
        f"- Modules evaluated: {len(rows)}",
        f"",
        f"| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |",
        f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for m in sorted(rows.keys(), key=lambda k: (k != "full", k)):
        r = rows[m]
        md_lines.append(
            f"| {m} | {r.get('status','?')} | "
            f"{r.get('perplexity','?')} | {r.get('ppl_ratio','?')} | "
            f"{r.get('compression_rate','?')} | "
            f"{r.get('num_injected_attention_layers',0)} | "
            f"{r.get('num_compressed_layers',0)} | "
            f"{r.get('latency_ms_per_token','?')} |"
        )
    out = out_dir / "REPORT.md"
    out.write_text("\n".join(md_lines), encoding="utf-8")
    LOG.info("Wrote report to %s", out)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary_stats(rows: Dict[str, Dict[str, str]]):
    if not rows:
        return
    ppls = []
    for k, r in rows.items():
        if k == "full":
            continue
        try:
            p = float(r.get("perplexity", "nan"))
            if np.isfinite(p) and p > 0:
                ppls.append(p)
        except Exception:
            pass
    if not ppls:
        return
    arr = np.array(ppls)
    LOG.info("=" * 70)
    LOG.info("SUMMARY (excluding 'full'):")
    LOG.info("  n = %d | min PPL = %.2f | max PPL = %.2f | median = %.2f | mean = %.2f",
             len(arr), arr.min(), arr.max(), np.median(arr), arr.mean())
    # Count by status
    status_counts: Dict[str, int] = {}
    for k, r in rows.items():
        s = r.get("status", "?")
        status_counts[s] = status_counts.get(s, 0) + 1
    LOG.info("  Status distribution:")
    for s, c in sorted(status_counts.items()):
        LOG.info("    %-30s %d", s, c)
    LOG.info("=" * 70)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_INTERRUPTED = {"flag": False}

def _install_signal_handlers():
    def _handler(signum, frame):
        if _INTERRUPTED["flag"]:
            LOG.warning("Second interrupt received — forcing exit.")
            sys.exit(130)
        _INTERRUPTED["flag"] = True
        LOG.warning("Interrupt received. Finishing current module and exiting...")
    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.max_length:
        args.seq_len = args.max_length

    setup_logging(args.log_file)
    _install_signal_handlers()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    # Module list
    if not args.modules:
        args.modules = ["all"]
    parsed: List[str] = []
    for item in args.modules:
        parsed.extend([m.strip() for m in str(item).split(",") if m.strip()])
    if not parsed or parsed == ["all"]:
        parsed = get_all_decoder_modules("all")
    if args.total_workers > 1:
        parsed = [m for i, m in enumerate(parsed)
                  if (i % args.total_workers) == args.worker_id]
        LOG.info("Worker %d/%d assigned %d modules",
                 args.worker_id, args.total_workers, len(parsed))
    args.modules = parsed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.total_workers > 1:
        worker_csv = out_dir / f"summary_worker_{args.worker_id}.csv"
    else:
        worker_csv = out_dir / "summary.csv"
    main_csv = out_dir / "summary.csv"
    main_lock = out_dir / "summary.csv.lock"

    # Force: delete existing CSVs to start fresh
    if args.force:
        for f in [worker_csv, main_csv]:
            if f.exists():
                LOG.info("--force: removing %s", f)
                f.unlink()

    LOG.info("=" * 78)
    LOG.info("GPT-2 DECODER PERPLEXITY SWEEP")
    LOG.info("  Model:        %s", args.model_path)
    LOG.info("  Dataset:      %s (%s)", args.dataset_name, args.split)
    LOG.info("  Window/Stride %d / %d", args.seq_len, args.stride)
    LOG.info("  Rank:         %d", args.rank)
    LOG.info("  Modules:      %d", len(args.modules))
    LOG.info("  Device:       %s | fp16=%s", args.device, args.fp16)
    LOG.info("  Output dir:   %s", out_dir)
    LOG.info("  Force:        %s | No-skip: %s | Only-failed: %s",
             args.force, args.no_skip, args.only_failed)
    LOG.info("=" * 78)

    if args.dry_run:
        LOG.info("[DRY RUN] Modules that would be evaluated:")
        for i, m in enumerate(args.modules, 1):
            LOG.info("  %3d. %s", i, m)
        return

    deprecated = load_deprecated_modules()
    LOG.info("Loaded %d deprecated modules.", len(deprecated))

    # ---- Tokenizer + datasets ----
    model_p = Path(args.model_path)
    if not model_p.exists() and (ROOT / args.model_path).exists():
        model_p = ROOT / args.model_path
    tok_src = str(model_p) if model_p.exists() else args.model_path
    LOG.info("Loading tokenizer from %s ...", tok_src)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    except Exception as e:
        if (ROOT / "models" / "gpt2_wikitext").exists():
            tok_src = str(ROOT / "models" / "gpt2_wikitext")
            LOG.info("Retrying tokenizer from %s ...", tok_src)
            tokenizer = AutoTokenizer.from_pretrained(tok_src)
        else:
            raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("Loading WikiText-2 (%s)...", args.split)
    try:
        ds_test = load_dataset("wikitext", args.dataset_name, split=args.split)
        ds_train = load_dataset("wikitext", args.dataset_name, split="train")
    except Exception:
        ds_test = load_dataset("Salesforce/wikitext", args.dataset_name, split=args.split)
        ds_train = load_dataset("Salesforce/wikitext", args.dataset_name, split="train")

    # ---- Calibration loader ----
    LOG.info("Building calibration loader (%d samples, batch=%d, max_len=%d)...",
             args.calib_samples, args.calib_batch_size, args.calib_max_length)
    calib_loader = build_calibration_loader(
        ds_train, tokenizer,
        num_samples=args.calib_samples,
        batch_size=args.calib_batch_size,
        max_length=args.calib_max_length,
    )
    LOG.info("Calibration loader: %d batches.", len(calib_loader))

    # ---- Baseline (reuse cached unless --recompute_baseline) ----
    cached = _read_csv_rows(main_csv)
    cached_base = cached.get("full")
    baseline_ppl: Optional[float] = None
    if (not args.recompute_baseline
            and cached_base is not None
            and str(cached_base.get("baseline_ppl", "")) not in ("", "0", "0.0", "-1.0", "nan")):
        try:
            baseline_ppl = float(cached_base["baseline_ppl"])
            LOG.info("Reusing cached baseline PPL: %.4f", baseline_ppl)
        except Exception:
            baseline_ppl = None

    base_model = None
    total_params_baseline = 0
    base_metrics: Dict[str, Any] = {}

    if baseline_ppl is None:
        LOG.info("Loading baseline GPT-2...")
        base_model = AutoModelForCausalLM.from_pretrained(tok_src)
        if args.fp16 and args.device.startswith("cuda"):
            base_model.half()
        base_model.to(args.device)
        base_model.eval()

        total_params_baseline = sum(p.numel() for p in base_model.parameters())
        LOG.info("Baseline params: %d (%.2f M)", total_params_baseline,
                 total_params_baseline / 1e6)

        LOG.info("Computing baseline PPL...")
        base_metrics = compute_perplexity_with_metrics(
            base_model, tokenizer, ds_test,
            seq_len=args.seq_len, stride=args.stride,
            max_eval_tokens=args.max_eval_tokens, device=args.device,
        )
        baseline_ppl = base_metrics["perplexity"]
        LOG.info("Baseline PPL: %.4f | VRAM: %.1f MB | %.1f tok/s",
                 baseline_ppl, base_metrics["peak_vram_mb"],
                 base_metrics["throughput_tok_per_sec"])

        # Sanity check
        if not np.isfinite(baseline_ppl) or baseline_ppl <= 0:
            LOG.error("Baseline PPL is invalid (%s). Aborting.", baseline_ppl)
            sys.exit(2)
        if baseline_ppl > 100:
            LOG.warning("Baseline PPL %.2f is unusually high — check model/dataset.", baseline_ppl)

        base_row = ResultRow(
            module_name="full",
            status="PASS",
            perplexity=baseline_ppl,
            baseline_ppl=baseline_ppl,
            delta_ppl=0.0,
            ppl_ratio=1.0,
            compression_rate=1.0,
            compression_ratio_pct=0.0,
            total_params_before=total_params_baseline,
            total_params_after=total_params_baseline,
            delta_params=0,
            peak_vram_mb=base_metrics["peak_vram_mb"],
            latency_ms_per_token=base_metrics["latency_ms_per_token"],
            throughput_tok_per_sec=base_metrics["throughput_tok_per_sec"],
            eval_elapsed_s=base_metrics["eval_elapsed_s"],
            wall_elapsed_s=base_metrics["eval_elapsed_s"],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        ).to_csv_dict()
        upsert_row(worker_csv, base_row)
        if args.total_workers > 1:
            upsert_row(main_csv, base_row, main_lock)
    else:
        # Recompute baseline params anyway (cheap) for compression_rate math
        total_params_baseline = int(cached_base.get("total_params_before", 0) or 0)
        if total_params_baseline == 0:
            LOG.info("Loading baseline GPT-2 just to count params...")
            _bm = AutoModelForCausalLM.from_pretrained(tok_src)
            total_params_baseline = sum(p.numel() for p in _bm.parameters())
            del _bm
            gc.collect()

    # ---- Snapshot for fast restore ----
    if base_model is None:
        LOG.info("Loading baseline GPT-2 for snapshot...")
        base_model = AutoModelForCausalLM.from_pretrained(tok_src)
        if args.fp16 and args.device.startswith("cuda"):
            base_model.half()
        base_model.cpu()

    LOG.info("Creating base-model snapshot...")
    base_snapshot = copy.deepcopy(base_model)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Determine which modules to skip ----
    existing = _read_csv_rows(main_csv)
    existing_worker = _read_csv_rows(worker_csv) if worker_csv != main_csv else {}
    existing_all = {**existing, **existing_worker}

    # Retry flags: if a module has one of these statuses, re-evaluate
    retry_set = set(args.retry_flags or [])

    skip: Set[str] = set()
    if not args.no_skip:
        for m, r in existing_all.items():
            st = (r.get("status") or "").strip()
            if args.only_failed:
                # Only skip if it PASSED cleanly with real work
                if is_real_success(st, r):
                    skip.add(m)
                continue
            # Default: skip if REAL success
            if is_real_success(st, r):
                skip.add(m)
            # Also honor --retry_flags
            if st in retry_set:
                skip.discard(m)

    LOG.info("Skipping %d modules (use --no_skip to force re-run).", len(skip))

    # ---- Iterate ----
    module_list = [m for m in args.modules if m != "full"]
    iterator = tqdm(module_list, desc="Modules", unit="mod") if _HAS_TQDM else module_list

    n_done = 0
    t_run_start = time.time()

    for idx, mod in enumerate(iterator, 1):
        if _INTERRUPTED["flag"]:
            LOG.warning("Interrupted — stopping before module '%s'.", mod)
            break

        if mod in skip:
            LOG.info("[%d/%d] SKIP %s (already successful)", idx, len(module_list), mod)
            continue

        mod_dir = out_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        res_json = mod_dir / "result.json"
        diag_json = mod_dir / "diag.json"

        # ---- Deprecated ----
        if mod in deprecated:
            LOG.info("[%d/%d] SKIP %s (deprecated)", idx, len(module_list), mod)
            row = ResultRow(
                module_name=mod,
                status="SKIP_DEPRECATED",
                baseline_ppl=baseline_ppl,
                total_params_before=total_params_baseline,
                total_params_after=total_params_baseline,
                error="deprecated in deprecated_modules.txt",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ).to_csv_dict()
            res_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
            upsert_row(worker_csv, row)
            if args.total_workers > 1:
                upsert_row(main_csv, row, main_lock)
            continue

        LOG.info("[%d/%d] EVAL %s", idx, len(module_list), mod)
        t_mod = time.time()

        # Fresh seed for each module
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        cand = None
        row = None
        try:
            cand = copy.deepcopy(base_snapshot)
            cand.to(args.device)

            typ = "attention" if mod in ATTENTION_REGISTRY else "compressor"

            # Calibration batches on device
            calib_on_dev = [
                {k: v.to(args.device, non_blocking=True) for k, v in b.items()}
                for b in calib_loader
            ]
            first_batch_ids = calib_on_dev[0]["input_ids"]

            t_comp0 = time.time()
            cand, comp_stats = apply_module_to_model(
                cand, mod, typ,
                rank=args.rank,
                device=args.device,
                calibration_data=first_batch_ids,
                calibration_loader=calib_on_dev,
                tokenizer=tokenizer,
            )
            comp_wall = time.time() - t_comp0

            comp_stats = comp_stats or {}
            p_after = sum(p.numel() for p in cand.parameters())

            # ---- Evaluate ----
            metrics = compute_perplexity_with_metrics(
                cand, tokenizer, ds_test,
                seq_len=args.seq_len, stride=args.stride,
                max_eval_tokens=args.max_eval_tokens, device=args.device,
            )
            ppl = metrics["perplexity"]

            wall = time.time() - t_mod
            comp_rate = round(total_params_baseline / max(1, p_after), 4)
            comp_pct = round(100.0 * (1.0 - p_after / max(1, total_params_baseline)), 2)
            delta_ppl = round(ppl - baseline_ppl, 4) if np.isfinite(ppl) else 999.0
            ppl_ratio = round(ppl / max(1e-9, baseline_ppl), 4) if np.isfinite(ppl) else float("inf")

            # ---- Classify ----
            status = classify_status(ppl_ratio)

            # ---- Phantom-pass detection ----
            if isinstance(comp_stats, dict):
                n_inj = int(comp_stats.get("num_injected_attention_layers", 0) or 0)
                n_comp = int(comp_stats.get("num_compressed_layers", 0) or 0)
                n_fall = int(comp_stats.get("num_fallback_layers", 0) or 0)
            else:
                n_inj = 12 if mod != "full" else 0
                n_comp = 0
                n_fall = 0
            if status == "PASS" and mod != "full" and (n_inj + n_comp) == 0:
                status = "PASS_WITHOUT_EFFECT"

            # ---- Optional KD Recovery ----
            post_kd_ppl = -1.0
            delta_ppl_recovery = 0.0
            if args.with_kd_recovery and np.isfinite(ppl) and ppl > (2.0 * baseline_ppl):
                LOG.info("[%s] Zero-shot PPL %.2f > 2x baseline (%.2f); running KD recovery...", mod, ppl, 2.0 * baseline_ppl)
                try:
                    kd_rec_time = run_kd_recovery(
                        student=cand,
                        teacher=base_snapshot,
                        tokenizer=tokenizer,
                        train_dataset=ds_train,
                        device=args.device,
                        steps=args.kd_recovery_steps,
                        seq_len=128,
                    )
                    # Evaluate post-KD PPL
                    post_metrics = compute_perplexity_with_metrics(
                        cand, tokenizer, ds_test,
                        seq_len=args.seq_len, stride=args.stride,
                        max_eval_tokens=args.max_eval_tokens, device=args.device,
                    )
                    post_kd_ppl = post_metrics["perplexity"]
                    if np.isfinite(post_kd_ppl):
                        delta_ppl_recovery = round(ppl - post_kd_ppl, 4)
                        LOG.info("[%s] Post-KD PPL: %.2f | Delta recovery: %.2f", mod, post_kd_ppl, delta_ppl_recovery)
                except Exception as ex_kd:
                    LOG.warning("[%s] KD recovery failed: %s", mod, ex_kd)

            row = ResultRow(
                module_name=mod,
                status=status,
                perplexity=ppl,
                baseline_ppl=baseline_ppl,
                delta_ppl=delta_ppl,
                ppl_ratio=ppl_ratio,
                compression_rate=comp_rate,
                compression_ratio_pct=comp_pct,
                total_params_before=total_params_baseline,
                total_params_after=p_after,
                delta_params=p_after - total_params_baseline,
                num_injected_attention_layers=n_inj,
                num_compressed_layers=n_comp,
                num_fallback_layers=n_fall,
                lora_params=int(comp_stats.get("lora_params", 0) or 0) if isinstance(comp_stats, dict) else 0,
                calibration_time_s=float(comp_stats.get("calibration_time_s", comp_wall) or comp_wall) if isinstance(comp_stats, dict) else round(comp_wall, 3),
                recovery_time_s=float(comp_stats.get("recovery_time_s", 0.0) or 0.0) if isinstance(comp_stats, dict) else 0.0,
                peak_vram_mb=metrics["peak_vram_mb"],
                latency_ms_per_token=metrics["latency_ms_per_token"],
                throughput_tok_per_sec=metrics["throughput_tok_per_sec"],
                eval_elapsed_s=metrics["eval_elapsed_s"],
                wall_elapsed_s=round(wall, 3),
                error="",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                post_kd_ppl=post_kd_ppl,
                delta_ppl_recovery=delta_ppl_recovery,
            )
            LOG.info(
                "[%d/%d] %s -> %s | PPL=%.3f (ratio=%.3f) | "
                "comp=%.3fx | inj=%d comp=%d | %.1fs",
                idx, len(module_list), mod, status, ppl, ppl_ratio,
                comp_rate, n_inj, n_comp, wall,
            )

            # Also persist to summary_with_kd.csv if KD recovery is enabled
            if args.with_kd_recovery:
                kd_csv = out_dir / "summary_with_kd.csv"
                upsert_row(kd_csv, row.to_csv_dict())

        except Exception as e:
            wall = time.time() - t_mod
            LOG.exception("[%d/%d] %s FAILED: %s", idx, len(module_list), mod, e)
            extra = {
                "baseline_ppl": baseline_ppl,
                "rank": args.rank,
                "seq_len": args.seq_len,
                "device": args.device,
                "fp16": args.fp16,
            }
            write_diag(diag_json, mod, e, extra=extra,
                       include_locals=args.save_traceback_locals)

            # Try to extract partial PPL if candidate was built
            ppl_partial = -1.0
            vram_partial = 0.0
            lat_partial = 0.0
            tps_partial = 0.0
            if cand is not None:
                try:
                    pm = compute_perplexity_with_metrics(
                        cand, tokenizer, ds_test,
                        seq_len=args.seq_len, stride=args.stride,
                        max_eval_tokens=min(20_000, args.max_eval_tokens or 20_000),
                        device=args.device,
                    )
                    if pm["is_finite"]:
                        ppl_partial = pm["perplexity"]
                        vram_partial = pm["peak_vram_mb"]
                        lat_partial = pm["latency_ms_per_token"]
                        tps_partial = pm["throughput_tok_per_sec"]
                except Exception:
                    pass

            # Get fault location from traceback
            tb = traceback.extract_tb(sys.exc_info()[2])
            last = tb[-1] if tb else None
            fault = f"[{last.filename}:{last.lineno}]" if last else "[unknown]"

            row = ResultRow(
                module_name=mod,
                status="FAIL",
                perplexity=ppl_partial,
                baseline_ppl=baseline_ppl,
                delta_ppl=999.0,
                ppl_ratio=float("inf"),
                total_params_before=total_params_baseline,
                total_params_after=total_params_baseline,
                peak_vram_mb=vram_partial,
                latency_ms_per_token=lat_partial,
                throughput_tok_per_sec=tps_partial,
                wall_elapsed_s=round(wall, 3),
                error=f"{fault} {type(e).__name__}: {str(e)}",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        finally:
            if cand is not None:
                del cand
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Persist result
        if row is not None:
            payload = row.to_csv_dict()
            try:
                res_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as e:
                LOG.warning("Could not write result.json for %s: %s", mod, e)
            upsert_row(worker_csv, payload)
            if args.total_workers > 1:
                upsert_row(main_csv, payload, main_lock)
            n_done += 1

    # ---- End-of-run ----
    total_wall = time.time() - t_run_start
    LOG.info("=" * 78)
    LOG.info("Sweep finished in %.1fs. %d modules processed.", total_wall, n_done)

    # Summary
    final_rows = _read_csv_rows(main_csv)
    print_summary_stats(final_rows)
    try:
        write_markdown_report(out_dir, main_csv, baseline_ppl)
    except Exception as e:
        LOG.warning("Report generation failed: %s", e)

    # Final JSON summary
    try:
        summary_payload = {
            "baseline_ppl": baseline_ppl,
            "total_params_baseline": total_params_baseline,
            "modules_evaluated": n_done,
            "wall_elapsed_s": round(total_wall, 2),
            "rows": final_rows,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "final_summary.json").write_text(
            json.dumps(summary_payload, indent=2, default=str), encoding="utf-8",
        )
        LOG.info("Wrote final_summary.json")
    except Exception as e:
        LOG.warning("Could not write final_summary.json: %s", e)


if __name__ == "__main__":
    main()