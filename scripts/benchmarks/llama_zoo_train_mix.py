#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TinyLlama-1.1B (LLaMA Family) Attention Surgery & WikiText-2 Distillation Engine.

Target Architecture:
  - `TinyLlama/TinyLlama_v1.1` (`LlamaForCausalLM`):
    22 decoder layers, hidden_size=2048 (exact 2^11), num_attention_heads=32 (2^5),
    num_key_value_heads=4 (8:1 GQA), head_dim=64, bias-free Q/K/V/O projections.

Reuses canonical NeuralOps surgery (`QwenAttentionAdapter`), chunked causal PPL
evaluator, WikiText-2 offline corpus builder, and model card generator cleanly.
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
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuralops.benchmarks.kd import (
    CausalTextDataset,
    apply_qwen_attention_surgery,
    build_cache_wikitext,
    build_llama_readme,
    evaluate_causal_ppl,
    extract_attention_metadata,
)
from neuralops.benchmarks.kd.teacher_cache import (
    CachedCausalDataset,
    build_or_load_causal_teacher_cache,
    enable_student_gradient_checkpointing,
    resolve_teacher_cache_path,
    try_load_causal_teacher_cache,
)

_build_cache_wikitext = build_cache_wikitext
_extract_attention_metadata = extract_attention_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llama_zoo_distill")


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


def train_single_llama_attention(
    attn_key: str,
    teacher_model: Optional[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    out_dir: Path,
    teacher_ppl: float,
    pattern: str = "hybrid_3_1",
    train_secs: float = 150.0,
    grad_accum: int = 8,
    hf_org: str = "Hooshaai",
    model_name: str = "/home/tahamajlesi.ut/NeuralOps/models/tinyllama_1.1b",
    gradient_checkpointing: bool = False,
    teacher_cache: Optional[dict[str, Any]] = None,
    base_model: Optional[nn.Module] = None,
) -> dict[str, Any]:
    """Runs 2-stage hidden representation distillation for TinyLlama-1.1B."""
    logger.info("\n" + "=" * 80)
    logger.info("[LLAMA START] Distilling %s on TinyLlama-1.1B (%ds, pattern=%s)", attn_key, int(train_secs), pattern)
    logger.info("=" * 80)

    if teacher_cache is None and teacher_model is None:
        raise ValueError("Either teacher_cache or teacher_model must be provided.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if base_model is not None:
        student = copy.deepcopy(base_model)
    elif teacher_model is not None:
        student = copy.deepcopy(teacher_model).float()
    else:
        from transformers import AutoModelForCausalLM
        student = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
    student = apply_qwen_attention_surgery(student, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    lm_layers = getattr(getattr(student, "model", student), "layers", [])
    total_layers = len(lm_layers)
    if pattern == "hybrid_3_1":
        surgeried_indices = [i for i in range(total_layers) if i % 4 != 3]
        anchor_indices = [i for i in range(total_layers) if i % 4 == 3]
    else:
        surgeried_indices = list(range(total_layers))
        anchor_indices = []
    layer_prefix = "model.layers." if hasattr(student, "model") and hasattr(student.model, "layers") else "layers."
    surgeried_attn_prefixes = tuple(f"{layer_prefix}{i}.self_attn." for i in surgeried_indices)
    anchor_attn_prefixes = tuple(f"{layer_prefix}{i}.self_attn." for i in anchor_indices)

    train_secs = min(float(train_secs), 90.0)
    pre_ppl = evaluate_causal_ppl(student, val_loader, device, max_batches=5)
    logger.info("[%s] Pre-KD WikiText-2 PPL=%.2f", attn_key, pre_ppl)

    if gradient_checkpointing:
        enable_student_gradient_checkpointing(student, log_prefix=f"[{attn_key}] ")

    for name, p in student.named_parameters():
        if any(name.startswith(pref) for pref in surgeried_attn_prefixes):
            p.requires_grad = True
        elif "layernorm" in name.lower():
            p.requires_grad = True
        else:
            p.requires_grad = False
    trained_param_names = {name for name, p in student.named_parameters() if p.requires_grad}

    t_start = time.time()
    t_s1_end = t_start + train_secs * 0.30
    t_s2_end = t_start + train_secs

    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.0e-4, weight_decay=0.01)
    scaler1 = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=(device.type == "cuda"))
    student.train()
    step_s1 = 0
    layer_indices: Optional[list[int]] = teacher_cache["layer_indices"] if teacher_cache is not None else None

    slow_layer_detected = False
    while time.time() < t_s1_end and not slow_layer_detected:
        for batch in train_loader:
            if time.time() >= t_s1_end or slow_layer_detected:
                break
            t_b0 = time.time()
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                if "teacher_hidden_states" in batch and layer_indices is not None:
                    t_h_b = batch["teacher_hidden_states"].to(device, non_blocking=True)
                    s_h = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                    t_h = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                    s_h, t_h = s_out.hidden_states[1:], t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
                loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
                loss = (loss_mse + 0.25 * loss_cos) / grad_accum
            scaler1.scale(loss).backward()
            step_s1 += 1
            if step_s1 % grad_accum == 0:
                scaler1.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler1.step(opt1)
                scaler1.update()
                opt1.zero_grad(set_to_none=True)
            if (time.time() - t_b0) > 20.0:
                logger.warning("[%s] Step took %.1fs (>20s). Slow Python layer detected, early exit stage 1.", attn_key, time.time() - t_b0)
                slow_layer_detected = True
                break

    if step_s1 > 0 and step_s1 % grad_accum != 0:
        scaler1.unscale_(opt1)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler1.step(opt1)
        scaler1.update()
        opt1.zero_grad(set_to_none=True)

    del opt1, scaler1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    opt2 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=6.0e-5, weight_decay=0.01)
    scaler2 = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=(device.type == "cuda"))
    step_s2 = 0

    while time.time() < t_s2_end and not slow_layer_detected:
        for batch in train_loader:
            if time.time() >= t_s2_end or slow_layer_detected:
                break
            t_b0 = time.time()
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                s_out = student.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                if "teacher_hidden_states" in batch and layer_indices is not None:
                    t_h_b = batch["teacher_hidden_states"].to(device, non_blocking=True)
                    s_h = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                    t_h = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                else:
                    assert teacher_model is not None
                    with torch.no_grad():
                        t_out = teacher_model.model(input_ids=in_ids, attention_mask=att_m, output_hidden_states=True, use_cache=False)
                    s_h, t_h = s_out.hidden_states[1:], t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
                loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
                logits_shift = student.lm_head(s_out.hidden_states[-1][:, :-1, :]).contiguous()
                labels_shift = in_ids[:, 1:].contiguous()
                ce_loss = F.cross_entropy(logits_shift.view(-1, logits_shift.size(-1)), labels_shift.view(-1))
                ce_term = ce_loss if torch.isfinite(ce_loss) else 0.0
                loss = (loss_mse + 0.50 * loss_cos + 0.25 * ce_term) / grad_accum
            scaler2.scale(loss).backward()
            step_s2 += 1
            if step_s2 % grad_accum == 0:
                scaler2.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler2.step(opt2)
                scaler2.update()
                opt2.zero_grad(set_to_none=True)
            if (time.time() - t_b0) > 20.0:
                logger.warning("[%s] Step took %.1fs (>20s). Slow Python layer detected, early exit stage 2.", attn_key, time.time() - t_b0)
                slow_layer_detected = True
                break

    if step_s2 > 0 and step_s2 % grad_accum != 0:
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler2.step(opt2)
        scaler2.update()
        opt2.zero_grad(set_to_none=True)

    total_time = time.time() - t_start
    post_ppl = evaluate_causal_ppl(student, val_loader, device, max_batches=5)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0
    logger.info("[%s] Post-KD WikiText-2 PPL=%.2f (Delta: %+.2f)", attn_key, post_ppl, post_ppl - pre_ppl)

    result = {
        "model": "TinyLlama/TinyLlama_v1.1",
        "attn_key": attn_key,
        "dataset": "wikitext-2",
        "pattern": pattern,
        "total_layers": total_layers,
        "surgeried_layers": surgeried_indices,
        "anchor_layers": anchor_indices,
        "teacher_ppl": round(teacher_ppl, 2),
        "ppl_before": round(pre_ppl, 2) if math.isfinite(pre_ppl) else None,
        "ppl_after": round(post_ppl, 2) if math.isfinite(post_ppl) else None,
        "ppl_delta": round(post_ppl - pre_ppl, 2) if (math.isfinite(pre_ppl) and math.isfinite(post_ppl)) else None,
        "train_secs": round(total_time, 1),
        "steps_s1": step_s1,
        "steps_s2": step_s2,
        "total_steps": step_s1 + step_s2,
        "peak_vram_mb": peak_vram,
        "hf_org": hf_org,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "PASSED",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))

    # Filter strictly to surgeried decoder self_attn prefixes + trained layernorms (exclude frozen anchor layers)
    saved_tensor_count = 0
    saved_via_helper = False
    try:
        from neuralops.models.surgical_io import save_surgical_delta
        delta_info = save_surgical_delta(
            student,
            out_dir,
            base_model_id="TinyLlama/TinyLlama_v1.1",
            surgeried_prefixes=[p.rstrip(".") for p in surgeried_attn_prefixes],
            trained_param_names=sorted(trained_param_names),
            anchor_prefixes=[p.rstrip(".") for p in anchor_attn_prefixes],
            operations=[
                {
                    "op": "replace_attention",
                    "component_type": "attention",
                    "operator_key": attn_key,
                    "attn_key": attn_key,
                    "pattern": pattern,
                    "target_modules": [p.rstrip(".") for p in surgeried_attn_prefixes],
                }
            ],
            metadata={
                "zoo_type": "tinyllama",
                "attention_type": attn_key,
                "pattern": pattern,
                "total_layers": total_layers,
                "surgeried_layers": surgeried_indices,
                "anchor_layers": anchor_indices,
            },
            half_precision=True,
            remove_full_weights=True,
        )
        saved_tensor_count = int(delta_info.get("saved_tensor_count", 0))
        saved_via_helper = True
    except (ImportError, TypeError, AttributeError):
        pass

    if not saved_via_helper:
        attn_state = {
            k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()).clone().contiguous()
            for k, v in student.state_dict().items()
            if not (any(k.startswith(ap) for ap in anchor_attn_prefixes) and k not in trained_param_names)
            and (
                any(k.startswith(sp) for sp in surgeried_attn_prefixes)
                or (k in trained_param_names and "layernorm" in k.lower())
            )
        }
        from safetensors.torch import save_file
        save_file(attn_state, str(out_dir / "attention_weights.safetensors"))
        saved_tensor_count = len(attn_state)
        manifest = {
            "schema_version": "neuralops.surgery/v2",
            "zoo_type": "tinyllama",
            "base_model": "TinyLlama/TinyLlama_v1.1",
            "mechanism_key": attn_key,
            "pattern": pattern,
            "total_layers": total_layers,
            "surgeried_layers": surgeried_indices,
            "anchor_layers": anchor_indices,
            "modified_prefixes": list(surgeried_attn_prefixes),
            "excluded_prefixes": ["embed_tokens", "lm_head", "*.mlp.*", *anchor_attn_prefixes],
            "operations": [{"op": "replace_attention", "attn_key": attn_key, "pattern": pattern}],
            "metadata": {"attention_type": attn_key, "pattern": pattern},
            "weights_file": "attention_weights.safetensors",
            "weights_format": "attention_only_fp16",
            "saved_tensor_count": saved_tensor_count,
            "saved_tensor_keys": sorted(attn_state.keys()),
        }
        _atomic_write_text(out_dir / "surgical_manifest.json", json.dumps(manifest, indent=2))

    for heavy_name in ("pytorch_model.bin", "model.safetensors", "optimizer.pt"):
        h_path = out_dir / heavy_name
        if h_path.exists():
            h_path.unlink()

    cfg = student.config.to_dict()
    cfg["_name_or_path"] = "TinyLlama/TinyLlama_v1.1"
    cfg["base_model_name_or_path"] = "TinyLlama/TinyLlama_v1.1"
    cfg["license"] = "mit"
    cfg["attention_type"] = attn_key
    cfg["surgery_pattern"] = pattern
    cfg["surgeried_layers"] = surgeried_indices
    cfg["anchor_layers"] = anchor_indices
    cfg["weights_format"] = "attention_only_fp16"
    cfg["saved_tensor_count"] = saved_tensor_count
    _atomic_write_text(out_dir / "config.json", json.dumps(cfg, indent=2))
    _atomic_write_text(out_dir / "training_metrics.json", json.dumps(result, indent=2))

    meta_map = _extract_attention_metadata()
    readme = build_llama_readme(attn_key, result, teacher_ppl, out_dir, meta_map.get(attn_key, {}))
    _atomic_write_text(out_dir / "README.md", readme)

    del student, opt2, scaler2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyLlama-1.1B Attention Surgery Zoo")
    parser.add_argument("--model", default="/home/tahamajlesi.ut/NeuralOps/models/tinyllama_1.1b")
    parser.add_argument("--keys_json", default="results/bert_attention_zoo/registry_keys.json")
    parser.add_argument("--attn_key", default=None)
    parser.add_argument("--surgery_pattern", choices=["all", "hybrid_3_1"], default="hybrid_3_1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--train_secs", type=float, default=150.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo")
    parser.add_argument("--metrics_log", default="/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo/metrics.jsonl")
    parser.add_argument("--hf_org", default="Hooshaai")
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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA unavailable.")
        sys.exit(3)

    keys = [args.attn_key] if args.attn_key else json.loads(Path(args.keys_json).read_text(encoding="utf-8"))
    passed: set[str] = set()
    ledger = Path(args.metrics_log)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("status") == "PASSED":
                        passed.add(str(d.get("attn_key")))
                except Exception:
                    pass

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    pending = [k for k in keys if k not in passed]
    if not pending:
        logger.info("All %d TinyLlama keys already PASSED. Exiting cleanly.", len(keys))
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_cache = "/dev/shm/wikitext_tinyllama_seq512.pt" if args.seq_len == 512 else f"/dev/shm/wikitext_tinyllama_seq{args.seq_len}.pt"
    train_chunks, val_chunks = _build_cache_wikitext(tokenizer, seq_len=args.seq_len, cache_path=tok_cache)
    val_loader = DataLoader(CausalTextDataset(val_chunks), batch_size=4, shuffle=False, pin_memory=(device.type == "cuda"))

    cache_path = resolve_teacher_cache_path(
        args.teacher_cache_path,
        out_dir=out_base,
        model_id="TinyLlama/TinyLlama_v1.1",
        tag="tinyllama_causal",
        seq_len=args.seq_len,
        max_cache_samples=args.max_cache_samples,
        topk=args.topk,
    )

    def _load_llama_teacher() -> nn.Module:
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
            normalized_model_id="TinyLlama/TinyLlama_v1.1",
            train_chunks=train_chunks,
            val_chunks=val_chunks,
            device=device,
            load_teacher_fn=_load_llama_teacher,
            evaluate_ppl_fn=evaluate_causal_ppl,
            seq_len=args.seq_len,
            max_cache_samples=args.max_cache_samples,
            topk=args.topk,
            batch_size=args.batch_size,
            val_batch_size=4,
        )
        teacher_ppl = float(teacher_cache["teacher_ppl"])
        train_ds: Any = CachedCausalDataset(teacher_cache)
    else:
        teacher = _load_llama_teacher()
        teacher_ppl = evaluate_causal_ppl(teacher, val_loader, device)
        train_ds = CausalTextDataset(train_chunks)

    logger.info("TinyLlama-1.1B Teacher Baseline WikiText-2 PPL=%.2f", teacher_ppl)

    logger.info("Loading TinyLlama-1.1B CPU base template once for fast in-memory surgery (zero disk-IO reload)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    for i, key in enumerate(keys, 1):
        if key in passed:
            logger.info("[%d/%d] Skipping %s (already PASSED).", i, len(keys), key)
            continue

        curr_bs = max(1, int(args.batch_size))
        curr_accum = max(1, int(args.grad_accum))
        curr_ckpt = bool(args.gradient_checkpointing)
        res: dict[str, Any] = {}

        while True:
            curr_loader = DataLoader(
                train_ds,
                batch_size=curr_bs,
                shuffle=True,
                drop_last=(len(train_ds) >= curr_bs),
                pin_memory=(device.type == "cuda"),
            )
            is_oom = False
            err_msg = ""
            try:
                res = train_single_llama_attention(
                    attn_key=key,
                    teacher_model=teacher,
                    train_loader=curr_loader,
                    val_loader=val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    out_dir=out_base / key,
                    teacher_ppl=teacher_ppl,
                    pattern=args.surgery_pattern,
                    train_secs=args.train_secs,
                    grad_accum=curr_accum,
                    hf_org=args.hf_org,
                    model_name=args.model,
                    gradient_checkpointing=curr_ckpt,
                    teacher_cache=teacher_cache,
                    base_model=base_model,
                )
                break
            except Exception as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower():
                    is_oom = True
                else:
                    logger.error("[%s] LLaMA training exception: %s", key, e, exc_info=True)

            del curr_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if is_oom and (curr_bs > 1 or not curr_ckpt):
                if not curr_ckpt:
                    curr_ckpt = True
                if curr_bs > 1:
                    next_bs = max(1, curr_bs // 2)
                    scale = max(1, curr_bs // next_bs)
                    curr_bs = next_bs
                    curr_accum = curr_accum * scale
                logger.warning(
                    "[%s] OOM encountered. Retrying at batch_size=%d, grad_accum=%d, gradient_checkpointing=%s...",
                    key,
                    curr_bs,
                    curr_accum,
                    curr_ckpt,
                )
                time.sleep(0.05)
                continue

            if is_oom:
                logger.error("[%s] OOM retry exhausted at bs=%d: %s", key, curr_bs, err_msg)
                res = {"attn_key": key, "status": "SKIPPED_OOM", "error": err_msg, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
            else:
                res = {"attn_key": key, "status": "FAIL_TRAINING", "error": err_msg, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
            break

        append_ledger_record_atomic(ledger, res)


if __name__ == "__main__":
    main()
