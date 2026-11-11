#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multimodal Vision-Language Attention Surgery & Visual Reasoning Distillation Engine.

Target Architecture:
  - `llava-hf/llava-interleave-qwen-0.5b-hf` (`LlavaForConditionalGeneration`):
    Frozen SigLIP Vision Tower + Multimodal Projector + 24-layer Qwen2 Decoder
    (hidden_size=1024, num_attention_heads=16, head_dim=64 — exact power-of-2 widths).

Custom Multimodal Benchmark Task (`derek-thomas/ScienceQA`):
  - Filtered image-bearing STEM & visual reasoning multiple-choice questions.
  - Visual patch tokens prepended before text tokens inside decoder self-attention.
  - Evaluates BOTH Image-Conditioned Answer Perplexity AND ScienceQA VQA Accuracy (%).
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
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_zoo_train_mix import (
    _extract_attention_metadata,
    apply_qwen_attention_surgery,
)
from neuralops.hub.model_card import generate_unified_model_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm_scienceqa_zoo")

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def _format_scienceqa_sample(sample: dict[str, Any]) -> tuple[str, str, int]:
    """Formats a ScienceQA sample into (prompt_without_answer, target_letter, choice_count)."""
    choices = sample.get("choices") or []
    hint = str(sample.get("hint") or "").strip()
    question = str(sample.get("question") or "").strip()
    ans_idx = int(sample.get("answer", 0))
    ans_idx = max(0, min(ans_idx, max(len(choices) - 1, 0)))

    lines = ["<image>", f"Question: {question}"]
    if hint:
        lines.append(f"Context: {hint}")
    lines.append("Choices:")
    for idx, ch in enumerate(choices[:5]):
        lines.append(f"({OPTION_LETTERS[idx]}) {ch}")
    lines.append("Answer with the option letter:")
    prompt = "\n".join(lines)
    letter = OPTION_LETTERS[ans_idx]
    return prompt, letter, min(len(choices), 5)


def _trim_image_tokens(
    enc: dict[str, torch.Tensor],
    image_token_id: int,
    expected_img_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure exact match between expanded image token placeholders and vision encoder output length."""
    ids = enc["input_ids"].squeeze(0)
    mask = enc["attention_mask"].squeeze(0)
    if expected_img_tokens > 0:
        img_pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
        extra = int(img_pos.numel()) - expected_img_tokens
        if extra > 0:
            drop_set = set(img_pos[-extra:].tolist())
            keep = torch.tensor([k not in drop_set for k in range(ids.size(0))], dtype=torch.bool)
            ids = ids[keep]
            mask = mask[keep]
    return ids, mask


def prepare_scienceqa_batches(
    data_dir: str,
    processor: Any,
    max_samples: int,
    max_seq_len: int = 896,
    image_token_id: int = 151646,
    expected_img_tokens: int = 729,
) -> list[dict[str, Any]]:
    """Pre-processes image+text ScienceQA pairs into compact CPU tensor examples with /dev/shm cache."""
    cache_file = Path(f"/dev/shm/scienceqa_{Path(data_dir).name}_n{max_samples}_img{expected_img_tokens}.pt")
    if cache_file.exists():
        try:
            examples = torch.load(cache_file, map_location="cpu")
            logger.info("Loaded %d cached ScienceQA examples from RAM: %s", len(examples), cache_file)
            return examples
        except Exception as e:
            logger.warning("Failed reading ScienceQA RAM cache %s (%s); rebuilding...", cache_file, e)

    from datasets import load_from_disk

    ds = load_from_disk(data_dir)
    n = min(len(ds), max_samples)
    examples: list[dict[str, Any]] = []
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in range(n):
        row = ds[i]
        img = row.get("image")
        if img is None:
            continue
        rgb = img.convert("RGB")
        prompt, letter, num_choices = _format_scienceqa_sample(row)
        full_text = f"{prompt} {letter}"

        enc_full = processor(text=full_text, images=rgb, return_tensors="pt", truncation=True, max_length=max_seq_len)
        enc_prompt = processor(text=prompt, images=rgb, return_tensors="pt", truncation=True, max_length=max_seq_len)

        input_ids, att_full = _trim_image_tokens(enc_full, image_token_id, expected_img_tokens)
        prompt_ids, att_prompt = _trim_image_tokens(enc_prompt, image_token_id, expected_img_tokens)
        if expected_img_tokens > 0 and int((input_ids == image_token_id).sum().item()) != expected_img_tokens:
            continue
        if expected_img_tokens > 0 and int((prompt_ids == image_token_id).sum().item()) != expected_img_tokens:
            continue
        prompt_len = int(prompt_ids.size(0))
        if input_ids.size(0) <= prompt_len:
            continue

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": att_full,
                "pixel_values": enc_full["pixel_values"].squeeze(0).half(),
                "labels": labels,
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": att_prompt,
                "answer_idx": OPTION_LETTERS.index(letter),
                "num_choices": max(num_choices, 2),
            }
        )

    logger.info("Prepared %d multimodal ScienceQA examples from %s", len(examples), data_dir)
    try:
        torch.save(examples, cache_file)
        logger.info("Saved ScienceQA RAM cache: %s", cache_file)
    except Exception as e:
        logger.debug("Could not write ScienceQA RAM cache: %s", e)
    return examples


def _collate_vlm_batch(batch: list[dict[str, Any]], pad_id: int, eval_prompt_only: bool = False) -> dict[str, torch.Tensor]:
    key_in = "prompt_input_ids" if eval_prompt_only else "input_ids"
    key_att = "prompt_attention_mask" if eval_prompt_only else "attention_mask"
    max_len = max(item[key_in].size(0) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    for idx, item in enumerate(batch):
        cur_len = item[key_in].size(0)
        input_ids[idx, :cur_len] = item[key_in]
        attention_mask[idx, :cur_len] = item[key_att]
        if not eval_prompt_only:
            labels[idx, :cur_len] = item["labels"]

    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
        "answer_idx": torch.tensor([item["answer_idx"] for item in batch], dtype=torch.long),
        "num_choices": torch.tensor([item["num_choices"] for item in batch], dtype=torch.long),
        "prompt_lens": torch.tensor([item["prompt_input_ids"].size(0) for item in batch], dtype=torch.long),
    }
    return out


def evaluate_vlm_scienceqa(
    model: nn.Module,
    val_examples: list[dict[str, Any]],
    processor: Any,
    device: torch.device,
    batch_size: int = 4,
    max_samples: int = 160,
) -> tuple[float, float]:
    """Computes BOTH (1) Image-Conditioned Answer PPL and (2) ScienceQA VQA Accuracy (%)."""
    model.eval()
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    letter_ids = [tokenizer.encode(f" {ch}", add_special_tokens=False)[-1] for ch in OPTION_LETTERS]

    subset = val_examples[:max_samples]
    total_nll = 0.0
    total_tok = 0
    correct = 0
    total_q = 0

    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            batch_items = subset[i : i + batch_size]
            batch = _collate_vlm_batch(batch_items, pad_id=pad_id, eval_prompt_only=False)
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            px = batch["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = model(input_ids=in_ids, attention_mask=att_m, pixel_values=px, use_cache=False)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = lbls[:, 1:].contiguous()
                valid_mask = shift_labels != -100
                if valid_mask.any():
                    loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    if torch.isfinite(loss_sum):
                        total_nll += float(loss_sum.item())
                        total_tok += int(valid_mask.sum().item())

            # Evaluate prompt completion logit over A..E
            p_batch = _collate_vlm_batch(batch_items, pad_id=pad_id, eval_prompt_only=True)
            p_ids = p_batch["input_ids"].to(device, non_blocking=True)
            p_att = p_batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                p_out = model(input_ids=p_ids, attention_mask=p_att, pixel_values=px, use_cache=False)
                p_logits = p_out.logits
                for b_idx in range(len(batch_items)):
                    pos = int(p_batch["prompt_lens"][b_idx].item()) - 1
                    n_ch = int(p_batch["num_choices"][b_idx].item())
                    cand_ids = letter_ids[:n_ch]
                    pred = int(p_logits[b_idx, pos, cand_ids].argmax().item())
                    if pred == int(p_batch["answer_idx"][b_idx].item()):
                        correct += 1
                    total_q += 1

    ppl = math.exp(min(total_nll / total_tok, 20.0)) if total_tok > 0 else float("inf")
    acc = (correct / max(total_q, 1)) * 100.0
    return acc, ppl


def build_vlm_readme(
    attn_key: str,
    result: dict[str, Any],
    teacher_acc: float,
    teacher_ppl: float,
    out_dir: Path,
    meta: dict[str, Any],
) -> str:
    """Generates a complete Hugging Face VLM Model Card via the unified NeuralOps generator."""
    slug = attn_key.replace("_", "-")
    repo_slug = f"llava-qwen-0.5b-{slug}-scienceqa"
    repo_id = f"{result.get('hf_org', 'tahamajs')}/{repo_slug}"
    pattern = str(result.get("pattern", "hybrid_3_1"))
    total_layers = int(result.get("total_layers", 24))
    surgeried_layers = result.get("surgeried_layers") or (
        [i for i in range(total_layers) if i % 4 != 3] if pattern == "hybrid_3_1" else list(range(total_layers))
    )
    anchor_layers = result.get("anchor_layers") or [i for i in range(total_layers) if i not in set(surgeried_layers)]

    merged_metrics = dict(result)
    merged_metrics["teacher_acc"] = teacher_acc
    if math.isfinite(teacher_ppl):
        merged_metrics["teacher_ppl"] = teacher_ppl

    surgery_info = {
        "pattern": pattern,
        "total_layers": total_layers,
        "modified_layers": len(surgeried_layers),
        "surgeried_layers": surgeried_layers,
        "anchor_layers": anchor_layers,
        "modified_components": (
            f"`{len(surgeried_layers)}/{total_layers}` decoder `language_model.model.layers.{{i}}.self_attn` "
            "+ trained decoder `layernorm`s"
        ),
        "frozen_components": (
            f"Frozen SigLIP `vision_tower.*` (27 layers), `multi_modal_projector.*`, "
            f"`{len(anchor_layers)}` quadratic anchor layers `{anchor_layers}`, decoder MLPs & embeddings"
        ),
        **({k: v for k, v in meta.items() if k in ("class_name", "family", "module_path", "docstring")}),
    }

    return generate_unified_model_card(
        zoo_type="llava_vlm",
        mechanism_key=attn_key,
        repo_id=repo_id,
        base_model="llava-hf/llava-interleave-qwen-0.5b-hf",
        out_dir=out_dir,
        metrics=merged_metrics,
        surgery_info=surgery_info,
        dataset="derek-thomas/ScienceQA",
    )


def train_single_vlm_attention(
    attn_key: str,
    teacher_vlm: nn.Module,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    processor: Any,
    device: torch.device,
    out_dir: Path,
    model_id: str,
    teacher_acc: float,
    teacher_ppl: float,
    pattern: str = "hybrid_3_1",
    train_secs: float = 120.0,
    batch_size: int = 4,
    grad_accum: int = 4,
    hf_org: str = "tahamajs",
) -> dict[str, Any]:
    """Runs 2-stage multimodal hidden+logit distillation for one transplanted VLM attention."""
    logger.info("\n" + "=" * 80)
    logger.info("[VLM START] Distilling %s on ScienceQA (%ds, bs=%d, pattern=%s)", attn_key, int(train_secs), batch_size, pattern)
    logger.info("=" * 80)

    pad_id = processor.tokenizer.pad_token_id or 0
    # Keep frozen vision_tower & projector in FP16 to save ~1 GB VRAM; train language_model in FP32 master weights
    student = copy.deepcopy(teacher_vlm)
    for comp_name in ("vision_tower", "multi_modal_projector"):
        comp = getattr(student, comp_name, None)
        if comp is not None:
            comp.half()
            for p in comp.parameters():
                p.requires_grad = False
    student.language_model = student.language_model.float()

    student.language_model = apply_qwen_attention_surgery(student.language_model, attn_key=attn_key, pattern=pattern)
    student = student.to(device)

    lm_layers = getattr(getattr(student.language_model, "model", student.language_model), "layers", [])
    total_layers = len(lm_layers)
    if pattern == "hybrid_3_1":
        surgeried_indices = [i for i in range(total_layers) if i % 4 != 3]
        anchor_indices = [i for i in range(total_layers) if i % 4 == 3]
    else:
        surgeried_indices = list(range(total_layers))
        anchor_indices = []
    lm_layer_prefix = (
        "language_model.model.layers."
        if hasattr(student.language_model, "model") and hasattr(student.language_model.model, "layers")
        else "language_model.layers."
    )
    surgeried_attn_prefixes = tuple(f"{lm_layer_prefix}{i}.self_attn." for i in surgeried_indices)
    anchor_attn_prefixes = tuple(f"{lm_layer_prefix}{i}.self_attn." for i in anchor_indices)

    pre_acc, pre_ppl = evaluate_vlm_scienceqa(student, val_examples, processor, device, batch_size=batch_size)
    logger.info("[%s] Pre-KD ScienceQA Acc=%.2f%% | Answer PPL=%.2f", attn_key, pre_acc, pre_ppl)

    for name, p in student.named_parameters():
        if name.startswith(("vision_tower.", "multi_modal_projector.")):
            p.requires_grad = False
        elif any(name.startswith(pref) for pref in surgeried_attn_prefixes):
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
    perm = torch.randperm(len(train_examples)).tolist()

    while time.time() < t_s1_end:
        for i in range(0, len(perm), batch_size):
            if time.time() >= t_s1_end:
                break
            b_items = [train_examples[j] for j in perm[i : i + batch_size]]
            batch = _collate_vlm_batch(b_items, pad_id=pad_id, eval_prompt_only=False)
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            px = batch["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_vlm(input_ids=in_ids, attention_mask=att_m, pixel_values=px, output_hidden_states=True, use_cache=False)
                s_out = student(input_ids=in_ids, attention_mask=att_m, pixel_values=px, output_hidden_states=True, use_cache=False)

                s_h = s_out.hidden_states[1:]
                t_h = t_out.hidden_states[1:]
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

    while time.time() < t_s2_end:
        for i in range(0, len(perm), batch_size):
            if time.time() >= t_s2_end:
                break
            b_items = [train_examples[j] for j in perm[i : i + batch_size]]
            batch = _collate_vlm_batch(b_items, pad_id=pad_id, eval_prompt_only=False)
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)
            px = batch["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                with torch.no_grad():
                    t_out = teacher_vlm(input_ids=in_ids, attention_mask=att_m, pixel_values=px, output_hidden_states=True, use_cache=False)
                s_out = student(input_ids=in_ids, attention_mask=att_m, pixel_values=px, labels=lbls, output_hidden_states=True, use_cache=False)

                s_h = s_out.hidden_states[1:]
                t_h = t_out.hidden_states[1:]
                loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
                loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
                ce_term = s_out.loss if (s_out.loss is not None and torch.isfinite(s_out.loss)) else 0.0
                loss = (loss_mse + 0.50 * loss_cos + 1.0 * ce_term) / grad_accum

            scaler2.scale(loss).backward()
            step_s2 += 1
            if step_s2 % grad_accum == 0:
                scaler2.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler2.step(opt2)
                scaler2.update()
                opt2.zero_grad(set_to_none=True)

    if step_s2 > 0 and step_s2 % grad_accum != 0:
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler2.step(opt2)
        scaler2.update()
        opt2.zero_grad(set_to_none=True)

    total_time = time.time() - t_start
    post_acc, post_ppl = evaluate_vlm_scienceqa(student, val_examples, processor, device, batch_size=batch_size)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0
    logger.info(
        "[%s] Post-KD ScienceQA Acc=%.2f%% (Delta %+.2f%%) | Answer PPL=%.2f (Delta %+.2f)",
        attn_key,
        post_acc,
        post_acc - pre_acc,
        post_ppl,
        post_ppl - pre_ppl,
    )

    result = {
        "model": "llava-hf/llava-interleave-qwen-0.5b-hf",
        "attn_key": attn_key,
        "dataset": "derek-thomas/ScienceQA",
        "pattern": pattern,
        "total_layers": total_layers,
        "surgeried_layers": surgeried_indices,
        "anchor_layers": anchor_indices,
        "teacher_acc": round(teacher_acc, 2),
        "acc_before": round(pre_acc, 2),
        "acc_after": round(post_acc, 2),
        "acc_delta": round(post_acc - pre_acc, 2),
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
    processor.save_pretrained(str(out_dir))

    # Filter strictly to surgeried decoder self_attn prefixes + trained layernorms (exclude frozen vision_tower & anchor layers)
    attn_state = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()).clone().contiguous()
        for k, v in student.state_dict().items()
        if not k.startswith(("vision_tower.", "multi_modal_projector."))
        and not (any(k.startswith(ap) for ap in anchor_attn_prefixes) and k not in trained_param_names)
        and (
            any(k.startswith(sp) for sp in surgeried_attn_prefixes)
            or (k in trained_param_names and "layernorm" in k.lower())
        )
    }

    saved_via_helper = False
    try:
        from neuralops.models.surgical_io import save_surgical_delta
        save_surgical_delta(
            student,
            out_dir,
            base_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
            surgeried_prefixes=[p.rstrip(".") for p in surgeried_attn_prefixes],
            trained_param_names=sorted(trained_param_names),
            anchor_prefixes=[p.rstrip(".") for p in anchor_attn_prefixes],
            operations=[{"op": "replace_attention", "attn_key": attn_key, "pattern": pattern}],
            metadata={
                "zoo_type": "llava_vlm",
                "attention_type": attn_key,
                "pattern": pattern,
                "total_layers": total_layers,
                "surgeried_layers": surgeried_indices,
                "anchor_layers": anchor_indices,
            },
        )
        saved_via_helper = True
    except (ImportError, TypeError, AttributeError):
        pass

    if not saved_via_helper:
        from safetensors.torch import save_file
        save_file(attn_state, str(out_dir / "attention_weights.safetensors"))
        manifest = {
            "schema_version": "neuralops.surgery/v2",
            "zoo_type": "llava_vlm",
            "base_model": "llava-hf/llava-interleave-qwen-0.5b-hf",
            "mechanism_key": attn_key,
            "pattern": pattern,
            "total_layers": total_layers,
            "surgeried_layers": surgeried_indices,
            "anchor_layers": anchor_indices,
            "modified_prefixes": list(surgeried_attn_prefixes),
            "excluded_prefixes": ["vision_tower.", "multi_modal_projector.", *anchor_attn_prefixes],
            "operations": [{"op": "replace_attention", "attn_key": attn_key, "pattern": pattern}],
            "metadata": {"attention_type": attn_key, "pattern": pattern},
            "weights_file": "attention_weights.safetensors",
            "weights_format": "attention_only_fp16",
            "saved_tensor_count": len(attn_state),
            "saved_tensor_keys": sorted(attn_state.keys()),
        }
        (out_dir / "surgical_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for heavy_name in ("pytorch_model.bin", "model.safetensors", "optimizer.pt"):
        h_path = out_dir / heavy_name
        if h_path.exists():
            h_path.unlink()

    cfg = student.config.to_dict()
    cfg["_name_or_path"] = "llava-hf/llava-interleave-qwen-0.5b-hf"
    cfg["base_model_name_or_path"] = "llava-hf/llava-interleave-qwen-0.5b-hf"
    cfg["license"] = "mit"
    cfg["attention_type"] = attn_key
    cfg["surgery_pattern"] = pattern
    cfg["surgeried_layers"] = surgeried_indices
    cfg["anchor_layers"] = anchor_indices
    cfg["weights_format"] = "attention_only_fp16"
    cfg["saved_tensor_count"] = len(attn_state)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    meta_map = _extract_attention_metadata()
    readme = build_vlm_readme(attn_key, result, teacher_acc, teacher_ppl, out_dir, meta_map.get(attn_key, {}))
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    del student, opt2, scaler2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaVA-Qwen-0.5B Multimodal Attention Surgery Zoo")
    parser.add_argument("--model", default="/home/tahamajlesi.ut/NeuralOps/models/llava_qwen_0.5b")
    parser.add_argument("--keys_json", default="results/bert_attention_zoo/registry_keys.json")
    parser.add_argument("--attn_key", default=None)
    parser.add_argument("--train_data", default="/home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_train")
    parser.add_argument("--val_data", default="/home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_val")
    parser.add_argument("--out_dir", default="/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo")
    parser.add_argument("--metrics_log", default="/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo/metrics.jsonl")
    parser.add_argument("--surgery_pattern", choices=["all", "hybrid_3_1"], default="hybrid_3_1")
    parser.add_argument("--train_secs", type=float, default=120.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=800)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA unavailable.")
        sys.exit(3)

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    teacher = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    img_tok_id = int(getattr(teacher.config, "image_token_index", 151646))
    vcfg = getattr(teacher.config, "vision_config", None)
    patch_sz = int(getattr(vcfg, "patch_size", 14) or 14)
    img_sz = int(getattr(vcfg, "image_size", 384) or 384)
    strategy = str(getattr(teacher.config, "vision_feature_select_strategy", "full"))
    expected_img_tokens = (img_sz // patch_sz) ** 2 + (0 if strategy == "full" else -1 if strategy == "default" else 0)
    logger.info("LLaVA vision token contract: image_token_index=%d, expected_img_tokens=%d (strategy=%s)", img_tok_id, expected_img_tokens, strategy)

    train_ex = prepare_scienceqa_batches(
        args.train_data, processor, max_samples=args.max_train_samples, image_token_id=img_tok_id, expected_img_tokens=expected_img_tokens
    )
    val_ex = prepare_scienceqa_batches(
        args.val_data, processor, max_samples=args.max_val_samples, image_token_id=img_tok_id, expected_img_tokens=expected_img_tokens
    )
    teacher_acc, teacher_ppl = evaluate_vlm_scienceqa(teacher, val_ex, processor, device, batch_size=args.batch_size)
    logger.info("Uncompressed VLM Teacher Baseline: ScienceQA Acc=%.2f%% | Answer PPL=%.2f", teacher_acc, teacher_ppl)

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
    for i, key in enumerate(keys, 1):
        if key in passed:
            logger.info("[%d/%d] Skipping %s (already PASSED).", i, len(keys), key)
            continue
        res = None
        err_msg = None
        is_oom = False
        try:
            res = train_single_vlm_attention(
                attn_key=key,
                teacher_vlm=teacher,
                train_examples=train_ex,
                val_examples=val_ex,
                processor=processor,
                device=device,
                out_dir=out_base / key,
                model_id=args.model,
                teacher_acc=teacher_acc,
                teacher_ppl=teacher_ppl,
                pattern=args.surgery_pattern,
                train_secs=args.train_secs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                hf_org=args.hf_org,
            )
        except Exception as e:
            err_msg = str(e)
            is_oom = "out of memory" in err_msg.lower()
            if not is_oom:
                logger.error("[%s] VLM training exception: %s", key, e, exc_info=True)

        if res is None:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if is_oom and args.batch_size > 1:
                retry_bs = max(1, args.batch_size // 2)
                logger.warning("[%s] OOM at bs=%d. Retrying at bs=%d...", key, args.batch_size, retry_bs)
                try:
                    res = train_single_vlm_attention(
                        attn_key=key,
                        teacher_vlm=teacher,
                        train_examples=train_ex,
                        val_examples=val_ex,
                        processor=processor,
                        device=device,
                        out_dir=out_base / key,
                        model_id=args.model,
                        teacher_acc=teacher_acc,
                        teacher_ppl=teacher_ppl,
                        pattern=args.surgery_pattern,
                        train_secs=args.train_secs,
                        batch_size=retry_bs,
                        grad_accum=args.grad_accum * 2,
                        hf_org=args.hf_org,
                    )
                except Exception as e2:
                    logger.error("[%s] VLM OOM retry exception: %s", key, e2)
                    res = {"attn_key": key, "status": "SKIPPED_OOM", "error": str(e2), "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                res = {"attn_key": key, "status": "FAIL_TRAINING", "error": err_msg, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
