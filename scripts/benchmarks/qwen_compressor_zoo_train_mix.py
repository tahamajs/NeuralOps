#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Master Track Qwen2.5-0.5B 82-Compressor Zoo Distillation Engine.

Compresses Qwen/Qwen2.5-0.5B using NeuralOps compressors from
results/compressor_zoo/registry_keys.json, measures exact parameter reduction
and sparsity, heals distortion via 2-stage Knowledge Distillation (Top-K Sparse
Logit KD + Causal CE + Layerwise Hidden-State MSE) from offline RAM caches, and
exports compact compressor_only_fp16 safetensors + Hugging Face Model Cards.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch 1.12 compatibility with modern HuggingFace Accelerate
try:
    _orig_named_parameters = nn.Module.named_parameters

    def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)

    nn.Module.named_parameters = _compat_named_parameters
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neuralops.core.compression_pipeline import compress_model
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor
from neuralops.hub.model_card import generate_unified_model_card, normalize_base_model_id
from neuralops.hub.uploader import AsyncHFHubUploader
from neuralops.models.surgical_io import extract_surgical_manifest
from neuralops.benchmarks.kd import (
    StratifiedQwenDataset,
    compute_topk_sparse_kd_loss,
    evaluate_causal_ppl,
    load_tokenized_qwen_corpus,
)
from neuralops.benchmarks.kd.teacher_cache import (
    CachedCausalDataset,
    build_or_load_causal_teacher_cache,
    compute_topk_sparse_kd_loss_from_cache,
    enable_student_gradient_checkpointing,
    resolve_teacher_cache_path,
)


def _normalize_base_model_id(model_name: str) -> str:
    return normalize_base_model_id(model_name, zoo_type="qwen_compressor")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_compressor_zoo_train")


def append_ledger_record_atomic(dest_path: Path | str, record: dict[str, Any]) -> None:
    """Atomically append a record to a JSONL ledger via a .tmp file and POSIX rename."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    if dest.exists():
        shutil.copyfile(dest, tmp_path)
    with tmp_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(dest)


def _atomic_write_text(dest_path: Path, content: str) -> None:
    """Atomically write text content to a destination file via a .tmp file and POSIX rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(dest_path)


def _compute_sparsity_pct(model: nn.Module) -> float:
    """Measure exact percentage of zero elements across floating weights and buffers."""
    total_elems = 0
    zero_elems = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point() and tensor.numel() > 0:
            total_elems += tensor.numel()
            zero_elems += int((tensor == 0).sum().item())
    if total_elems == 0:
        return 0.0
    return round((zero_elems / total_elems) * 100.0, 2)


_DELTA_KEYWORDS = (
    "compress",
    "lora",
    "adapter",
    "factor",
    "svd",
    "quant",
    "mask",
    "scale",
    "zero",
    "u_proj",
    "v_proj",
    "weight_u",
    "weight_v",
    "core",
    "codebook",
    "centroid",
)


def _save_compressed_weights(
    student: nn.Module,
    out_dir: Path,
    *,
    comp_key: str = "",
    rank: int = 128,
    base_model: str = "Qwen/Qwen2.5-0.5B",
    compressed_prefixes: tuple[str, ...] = (),
    trained_param_names: set[str] | None = None,
) -> tuple[int, list[str]]:
    """Persist ONLY compressed / modified / trained delta parameters in compact FP16 safetensors."""
    trained_set = trained_param_names or set()
    comp_state: dict[str, torch.Tensor] = {}
    saved_prefixes: set[str] = set()

    for k, v in student.state_dict().items():
        k_lower = k.lower()
        in_compressed = any(k.startswith(pref) for pref in compressed_prefixes)
        has_kw = any(term in k_lower for term in _DELTA_KEYWORDS)
        is_trained = k in trained_set
        if in_compressed or has_kw or is_trained:
            comp_state[k] = (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()).clone().contiguous()
            if "." in k:
                saved_prefixes.add(k.rsplit(".", 1)[0])

    sorted_prefixes = sorted(saved_prefixes)
    from safetensors.torch import save_file
    save_file(comp_state, str(out_dir / "attention_weights.safetensors"))
    manifest = extract_surgical_manifest(student)
    manifest.update(
        {
            "schema_version": "neuralops.surgery/v2",
            "zoo_type": "qwen_compressor",
            "base_model": _normalize_base_model_id(base_model),
            "mechanism_key": comp_key,
            "pattern": f"compress_model(method={comp_key}, rank={rank})",
            "rank": rank,
            "total_layers": int(getattr(student.config, "num_hidden_layers", 24)),
            "modified_prefixes": sorted_prefixes,
            "excluded_prefixes": ["model.embed_tokens.*", "lm_head.* (when uncompressed)"],
            "operations": [{"op": "compress_model", "method": comp_key, "rank": rank}],
            "metadata": {"compression_method": comp_key, "rank": rank},
            "weights_file": "attention_weights.safetensors",
            "weights_format": "compressor_only_fp16",
            "saved_tensor_count": len(comp_state),
            "saved_tensor_keys": sorted(comp_state.keys()),
        }
    )
    _atomic_write_text(out_dir / "surgical_manifest.json", json.dumps(manifest, indent=2))

    for heavy_name in ["pytorch_model.bin", "model.safetensors", "optimizer.pt"]:
        h_path = out_dir / heavy_name
        if h_path.exists():
            os.remove(h_path)

    return len(comp_state), sorted_prefixes


def _build_qwen_comp_readme(
    *,
    repo_id: str,
    base_model: str,
    comp_key: str,
    rank: int,
    orig_params: int,
    comp_params: int,
    param_reduction: float,
    sparsity_pct: float,
    pre_ppl: float | None,
    post_ppl: float | None,
    steps_s1: int,
    steps_s2: int,
    train_secs: float,
    peak_vram_mb: int,
    out_dir: Path,
    modified_prefixes: list[str] | None = None,
) -> str:
    """Construct a complete Hugging Face Model Card via the unified NeuralOps generator."""
    metrics = {
        "orig_params": orig_params,
        "comp_params": comp_params,
        "param_reduction": param_reduction,
        "compression_ratio": param_reduction,
        "sparsity_pct": sparsity_pct,
        "ppl_before": pre_ppl,
        "ppl_after": post_ppl,
        "steps_s1": steps_s1,
        "steps_s2": steps_s2,
        "total_steps": steps_s1 + steps_s2,
        "train_secs": train_secs,
        "peak_vram_mb": peak_vram_mb,
        "rank": rank,
    }
    surgery_info = {
        "pattern": f"compress_model(method={comp_key}, rank={rank})",
        "rank": rank,
        "total_layers": 24,
        "modified_prefixes": modified_prefixes or [],
    }
    return generate_unified_model_card(
        zoo_type="qwen_compressor",
        mechanism_key=comp_key,
        repo_id=repo_id,
        base_model=_normalize_base_model_id(base_model),
        out_dir=out_dir,
        metrics=metrics,
        surgery_info=surgery_info,
        dataset="Salesforce/wikitext",
    )


def train_single_qwen_compressor(
    comp_key: str,
    teacher_model: Optional[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    repo_id: str = "tahamajs/qwen-2.5-0.5b-comp-unknown-wikitext2",
    train_secs: float = 60.0,
    topk: int = 64,
    rank: int = 128,
    grad_accum: int = 8,
    gradient_checkpointing: bool = False,
    teacher_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply `comp_key` compression on CPU FP32, then heal on GPU using AMP + Top-K Sparse KD."""
    t_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("[START] Distilling Qwen Compressor: %s (%.1fs, rank=%d)", comp_key, train_secs, rank)
    logger.info("=" * 80)

    if teacher_cache is None and teacher_model is None:
        raise ValueError("Either teacher_cache or teacher_model must be provided.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Instantiate fresh FP32 base on CPU so SVD/QR/decomposition compressors run cleanly
    # and trainable master parameters stay FP32 for PyTorch 1.12 GradScaler compatibility.
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    orig_params = sum(p.numel() for p in student.parameters())
    base_shapes = {k: tuple(v.shape) for k, v in student.state_dict().items()}
    base_fp16_sd = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in student.state_dict().items()
    }

    try:
        cfg = {"compressors": [{"method": comp_key, "rank": rank, "force_rank": True, "use_strs": False}]}
        student = compress_model(student, cfg)
    except Exception as e:
        logger.warning("[%s] compress_model failed, falling back to factory: %s", comp_key, e)
        inst = create_compressor(comp_key, model=student, rank=rank)
        if hasattr(inst, "compress"):
            res = inst.compress()
            student = res[0] if isinstance(res, tuple) else res

    # Identify strictly compressed/modified prefixes by comparing post-compression state vs base snapshot.
    compressed_prefix_set: set[str] = set()
    for k, v in student.state_dict().items():
        k_lower = k.lower()
        t_v = base_fp16_sd.get(k)
        if t_v is None or tuple(v.shape) != base_shapes.get(k):
            is_changed = True
        elif any(term in k_lower for term in _DELTA_KEYWORDS):
            is_changed = True
        elif v.is_floating_point() and t_v.is_floating_point():
            is_changed = not torch.equal(v.detach().cpu().half(), t_v)
        else:
            is_changed = v.dtype != t_v.dtype or not torch.equal(v.detach().cpu(), t_v)
        if is_changed and "." in k:
            compressed_prefix_set.add(k.rsplit(".", 1)[0] + ".")
    compressed_prefixes = tuple(sorted(compressed_prefix_set))
    del base_fp16_sd, base_shapes

    # Normalize any compressed float16 buffers back to FP32 if trainable so GradScaler works safely
    for p in student.parameters():
        if p.dtype == torch.float16:
            p.data = p.data.float()

    student = student.to(device)
    comp_params = sum(p.numel() for p in student.parameters())
    param_reduction = orig_params / max(comp_params, 1)
    sparsity_pct = _compute_sparsity_pct(student)
    logger.info(
        "[%s] Params: %d -> %d (Ratio=%.3fx | Sparsity=%.2f%%)",
        comp_key, orig_params, comp_params, param_reduction, sparsity_pct,
    )

    pre_ppl = evaluate_causal_ppl(student, val_loader, device)
    logger.info("[%s] Pre-KD WikiText-2 PPL: %.2f", comp_key, pre_ppl)

    if gradient_checkpointing:
        enable_student_gradient_checkpointing(student, log_prefix=f"[{comp_key}] ")

    t_train_start = time.time()
    t_s1_deadline = t_train_start + (train_secs * 0.30)
    t_s2_deadline = t_train_start + train_secs

    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=(device.type == "cuda"))

    # Stage 1: Frozen Layer Alignment (compressed/attn/ln parameters)
    trainable_s1 = []
    trained_param_names: set[str] = set()
    for name, p in student.named_parameters():
        is_target = any(k in name.lower() for k in ("compress", "svd", "factor", "lora", "u_proj", "v_proj", "self_attn", "layernorm"))
        p.requires_grad = bool(is_target and p.is_floating_point())
        if p.requires_grad:
            trainable_s1.append(p)
            trained_param_names.add(name)
    if not trainable_s1:
        for name, p in student.named_parameters():
            if p.is_floating_point():
                p.requires_grad = True
                trainable_s1.append(p)
                trained_param_names.add(name)

    opt1 = torch.optim.AdamW(trainable_s1, lr=8.0e-4, weight_decay=0.01)
    student.train()
    step_s1 = 0
    layer_indices: Optional[list[int]] = teacher_cache["layer_indices"] if teacher_cache is not None else None

    while time.time() < t_s1_deadline:
        for batch in train_loader:
            if time.time() >= t_s1_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                s_out = student(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                if "teacher_hidden_states" in batch and layer_indices is not None:
                    t_h_b = batch["teacher_hidden_states"].to(device, non_blocking=True)
                    s_h = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                    t_h = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                    s_h = s_out.hidden_states[1:]
                    t_h = t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(sh.float(), th.float()) for sh, th in zip(s_h, t_h)]).mean()
                loss = loss_mse / grad_accum
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            step_s1 += 1
            if step_s1 % grad_accum == 0:
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(trainable_s1, 1.0)
                scaler.step(opt1)
                scaler.update()
                opt1.zero_grad(set_to_none=True)

    # Flush Stage 1 leftovers
    if step_s1 % grad_accum != 0:
        scaler.unscale_(opt1)
        scaler.step(opt1)
        scaler.update()
        opt1.zero_grad(set_to_none=True)
    del opt1

    # Stage 2: End-to-End Compressed/Decoder Healing (Top-K Sparse Logit KD + Causal LM CE)
    # Keep untouched embed_tokens & lm_head frozen unless compressed to avoid bloat & train/save mismatch
    trainable_s2 = []
    for name, p in student.named_parameters():
        in_compressed = any(name.startswith(pref) for pref in compressed_prefixes)
        is_decoder_target = (".self_attn." in name or ".mlp." in name or "layernorm" in name.lower()) and not name.startswith(("model.embed_tokens.", "lm_head."))
        should_train = p.is_floating_point() and (in_compressed or is_decoder_target)
        p.requires_grad = bool(should_train)
        if p.requires_grad:
            trainable_s2.append(p)
            trained_param_names.add(name)
    if not trainable_s2:
        for name, p in student.named_parameters():
            if p.is_floating_point():
                p.requires_grad = True
                trainable_s2.append(p)
                trained_param_names.add(name)

    opt2 = torch.optim.AdamW(trainable_s2, lr=1.0e-4, weight_decay=0.01)
    student.train()
    step_s2 = 0

    while time.time() < t_s2_deadline:
        for batch in train_loader:
            if time.time() >= t_s2_deadline:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                s_out = student(input_ids=in_ids, attention_mask=att_m, labels=lbls, use_cache=False)
                if "teacher_topk_vals" in batch and "teacher_topk_idx" in batch:
                    t_vals = batch["teacher_topk_vals"].to(device, non_blocking=True)
                    t_idx = batch["teacher_topk_idx"].to(device, non_blocking=True)
                    loss_sparse = compute_topk_sparse_kd_loss_from_cache(s_out.logits, t_vals, t_idx)
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model(input_ids=in_ids, attention_mask=att_m, use_cache=False)
                    loss_sparse = compute_topk_sparse_kd_loss(s_out.logits.float(), t_out.logits.float(), topk=topk)
                loss_ce = s_out.loss.float() if s_out.loss is not None else torch.tensor(0.0, device=device)
                loss = (0.50 * loss_sparse + 0.50 * loss_ce) / grad_accum
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            step_s2 += 1
            if step_s2 % grad_accum == 0:
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(trainable_s2, 1.0)
                scaler.step(opt2)
                scaler.update()
                opt2.zero_grad(set_to_none=True)

    if step_s2 % grad_accum != 0:
        scaler.unscale_(opt2)
        scaler.step(opt2)
        scaler.update()
        opt2.zero_grad(set_to_none=True)
    del opt2

    total_time = time.time() - t_start
    post_ppl = evaluate_causal_ppl(student, val_loader, device)
    peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0
    final_sparsity = _compute_sparsity_pct(student)

    is_finite_ok = math.isfinite(post_ppl)
    pre_val = round(pre_ppl, 2) if math.isfinite(pre_ppl) else None
    post_val = round(post_ppl, 2) if is_finite_ok else None
    delta_val = round(post_ppl - pre_ppl, 2) if (is_finite_ok and math.isfinite(pre_ppl)) else None
    status = "PASSED" if is_finite_ok else "FAIL_NUMERICS"

    result: dict[str, Any] = {
        "model": _normalize_base_model_id(model_name),
        "compressor": comp_key,
        "rank": rank,
        "param_reduction": round(param_reduction, 3),
        "sparsity_pct": final_sparsity,
        "orig_params": orig_params,
        "comp_params": comp_params,
        "ppl_before": pre_val,
        "ppl_after": post_val,
        "ppl_delta": delta_val,
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "peak_vram_mb": peak_vram,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": status,
    }

    if status == "PASSED":
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        num_tensors, saved_prefixes = _save_compressed_weights(
            student,
            out_dir,
            comp_key=comp_key,
            rank=rank,
            base_model=model_name,
            compressed_prefixes=compressed_prefixes,
            trained_param_names=trained_param_names,
        )

        cfg_out = student.config.to_dict()
        cfg_out["_name_or_path"] = _normalize_base_model_id(model_name)
        cfg_out["base_model_name_or_path"] = _normalize_base_model_id(model_name)
        cfg_out["license"] = "mit"
        cfg_out["compression_method"] = comp_key
        cfg_out["compressor_key"] = comp_key
        cfg_out["compression_rank"] = rank
        cfg_out["compression_ratio"] = round(param_reduction, 3)
        cfg_out["weights_format"] = "compressor_only_fp16"
        cfg_out["saved_tensor_count"] = num_tensors
        _atomic_write_text(out_dir / "config.json", json.dumps(cfg_out, indent=2))
        _atomic_write_text(out_dir / "training_metrics.json", json.dumps(result, indent=2))

        readme_text = _build_qwen_comp_readme(
            repo_id=repo_id,
            base_model=model_name,
            comp_key=comp_key,
            rank=rank,
            orig_params=orig_params,
            comp_params=comp_params,
            param_reduction=param_reduction,
            sparsity_pct=final_sparsity,
            pre_ppl=pre_val,
            post_ppl=post_val,
            steps_s1=step_s1,
            steps_s2=step_s2,
            train_secs=total_time,
            peak_vram_mb=peak_vram,
            out_dir=out_dir,
            modified_prefixes=saved_prefixes,
        )
        _atomic_write_text(out_dir / "README.md", readme_text)
        logger.info("[%s] Saved %d compressed tensors + README to %s (PPL: %s -> %s)", comp_key, num_tensors, out_dir, pre_val, post_val)

    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Qwen2.5-0.5B 82-Compressor Zoo Distillation")
    parser.add_argument("--keys_json", default="results/compressor_zoo/registry_keys.json")
    parser.add_argument("--comp_key", default="", help="Optional single compressor key for process isolation.")
    parser.add_argument("--model", default="/dev/shm/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--train_secs", type=float, default=60.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/qwen_compressor_zoo")
    parser.add_argument("--metrics_log", default="/home/tahamajlesi.ut/NeuralOps/results/qwen_compressor_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--hf_upload", type=int, default=0)
    parser.add_argument("--keep_local_weights", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="Enable gradient checkpointing on student decoder blocks (0/1)",
    )
    parser.add_argument(
        "--teacher_cache_path",
        default="auto",
        help="Offline teacher representation cache path ('auto' or 'none' to disable)",
    )
    parser.add_argument(
        "--max_cache_samples",
        type=int,
        default=256,
        help="Maximum training chunks/samples stored in offline teacher cache",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available.")
        sys.exit(3)

    if args.comp_key:
        comp_keys = [args.comp_key]
    else:
        keys_path = Path(args.keys_json)
        if not keys_path.exists():
            logger.error("Keys JSON not found: %s", keys_path)
            sys.exit(2)
        comp_keys = json.loads(keys_path.read_text(encoding="utf-8"))

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(args.metrics_log)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    done_passed: set[str] = set()
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "PASSED" and rec.get("compressor"):
                        done_passed.add(rec["compressor"])
                except Exception:
                    pass

    pending_keys = [
        k for k in comp_keys
        if k not in done_passed and not (out_base / k / "training_metrics.json").exists()
    ]
    if not pending_keys:
        logger.info("All requested Qwen compressor keys already completed. Exiting cleanly.")
        os._exit(0)

    hf_token = os.environ.get("HF_TOKEN")
    uploader = None
    if args.hf_upload and hf_token:
        uploader = AsyncHFHubUploader(token=hf_token, org=args.hf_org)
        uploader.start()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_chunks, val_chunks = load_tokenized_qwen_corpus(
        tokenizer,
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        cache_path_override=f"/tmp/wikitext_qwen_comp_seq{args.seq_len}.pt",
    )
    val_ds = StratifiedQwenDataset(val_chunks)

    norm_model_id = _normalize_base_model_id(args.model)
    cache_path = resolve_teacher_cache_path(
        args.teacher_cache_path,
        out_dir=out_base,
        model_id=norm_model_id,
        tag="qwen_causal",
        seq_len=args.seq_len,
        max_cache_samples=args.max_cache_samples,
        topk=args.topk,
    )

    def _load_qwen_teacher() -> nn.Module:
        t = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            local_files_only=True,
        ).to(device)
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        return t

    teacher: Optional[nn.Module] = None
    teacher_cache: Optional[dict[str, Any]] = None
    if cache_path is not None:
        teacher_cache = build_or_load_causal_teacher_cache(
            cache_path,
            model_id=args.model,
            normalized_model_id=norm_model_id,
            train_chunks=train_chunks,
            val_chunks=val_chunks,
            device=device,
            load_teacher_fn=_load_qwen_teacher,
            evaluate_ppl_fn=evaluate_causal_ppl,
            seq_len=args.seq_len,
            max_cache_samples=args.max_cache_samples,
            topk=args.topk,
            batch_size=args.batch_size,
            val_batch_size=args.batch_size,
        )
        train_ds: Any = CachedCausalDataset(teacher_cache)
    else:
        teacher = _load_qwen_teacher()
        train_ds = StratifiedQwenDataset(train_chunks)

    for i, comp_key in enumerate(pending_keys):
        logger.info("\n[%d/%d] Processing Qwen Compressor: %s", i + 1, len(pending_keys), comp_key)
        ckpt_dir = out_base / comp_key
        repo_id = f"{args.hf_org}/qwen-2.5-0.5b-comp-{comp_key.replace('_', '-')}-wikitext2"

        curr_bs = max(1, int(args.batch_size))
        curr_accum = max(1, int(args.grad_accum))
        curr_ckpt = bool(args.gradient_checkpointing)
        res: dict[str, Any] | None = None

        while True:
            curr_train_loader = DataLoader(
                train_ds,
                batch_size=curr_bs,
                shuffle=True,
                drop_last=(len(train_ds) >= curr_bs),
                pin_memory=(device.type == "cuda"),
            )
            curr_val_loader = DataLoader(val_ds, batch_size=curr_bs, shuffle=False, pin_memory=(device.type == "cuda"))
            is_oom = False
            err_msg = ""
            try:
                res = train_single_qwen_compressor(
                    comp_key=comp_key,
                    teacher_model=teacher,
                    train_loader=curr_train_loader,
                    val_loader=curr_val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    out_dir=ckpt_dir,
                    model_name=args.model,
                    repo_id=repo_id,
                    train_secs=args.train_secs,
                    topk=args.topk,
                    rank=args.rank,
                    grad_accum=curr_accum,
                    gradient_checkpointing=curr_ckpt,
                    teacher_cache=teacher_cache,
                )
                break
            except Exception as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower():
                    is_oom = True
                else:
                    logger.error("[%s] Training failed: %s", comp_key, e, exc_info=True)

            del curr_train_loader, curr_val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if is_oom and (curr_bs > 1 or not curr_ckpt):
                if not curr_ckpt:
                    curr_ckpt = True
                if curr_bs > 1:
                    next_bs = max(1, curr_bs // 2)
                    scale = max(1, curr_bs // next_bs)
                    curr_bs = next_bs
                    curr_accum = curr_accum * scale
                logger.warning(
                    "[%s] CUDA OOM -> retrying at batch_size=%d, grad_accum=%d, gradient_checkpointing=%s",
                    comp_key,
                    curr_bs,
                    curr_accum,
                    curr_ckpt,
                )
                time.sleep(0.05)
                continue

            if is_oom:
                logger.error("[%s] OOM retry exhausted at bs=%d: %s", comp_key, curr_bs, err_msg)
                res = {
                    "model": _normalize_base_model_id(args.model),
                    "compressor": comp_key,
                    "status": "SKIPPED_OOM",
                    "error": err_msg[:240],
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            else:
                res = {
                    "model": _normalize_base_model_id(args.model),
                    "compressor": comp_key,
                    "status": "FAIL_TRAINING",
                    "error": err_msg[:240],
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            break

        if res is not None:
            append_ledger_record_atomic(metrics_file, res)

            if res.get("status") == "PASSED" and args.hf_upload and uploader:
                uploader.submit_upload(
                    repo_id=repo_id,
                    folder_path=str(ckpt_dir),
                    delete_after_upload=bool(not args.keep_local_weights),
                )

    if uploader:
        logger.info("Waiting for background HF uploads...")
        uploader.shutdown(wait=True)

    logger.info("Qwen Compressor Zoo run finished.")
    os._exit(0)


if __name__ == "__main__":
    main()
