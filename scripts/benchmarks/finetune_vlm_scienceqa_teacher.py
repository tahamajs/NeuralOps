#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fine-Tune LLaVA-Qwen-0.5B Teacher on ScienceQA (Image Subset) Prior to VLM Zoo Distillation.

Mirrors the BERT SST-2 Teacher preparation protocol: raises the uncompressed
teacher accuracy from zero-shot (~46.9%) to a strong fine-tuned reference ceiling
(~75-85%+) before distilling all 110 subquadratic attention variants.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from vlm_llava_scienceqa_zoo import (
    _collate_vlm_batch,
    evaluate_vlm_scienceqa,
    prepare_scienceqa_batches,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_vlm_teacher")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Qwen-0.5B teacher on ScienceQA.")
    parser.add_argument("--model", default="/home/tahamajlesi.ut/NeuralOps/models/llava_qwen_0.5b")
    parser.add_argument("--train_data", default="/home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_train")
    parser.add_argument("--val_data", default="/home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_val")
    parser.add_argument("--out_dir", default="/tmp/llava_scienceqa_teacher")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--max_train_samples", type=int, default=1500)
    parser.add_argument("--max_val_samples", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA unavailable.")
        sys.exit(3)

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    # Keep frozen SigLIP vision tower in FP16 to preserve VRAM; fine-tune projector + Qwen2 decoder in FP32 + AMP
    if getattr(model, "vision_tower", None) is not None:
        model.vision_tower.half()
        for p in model.vision_tower.parameters():
            p.requires_grad = False
    if getattr(model, "multi_modal_projector", None) is not None:
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = True
    for p in model.language_model.parameters():
        p.requires_grad = True
    model = model.to(device)

    img_tok_id = int(getattr(model.config, "image_token_index", 151646))
    vcfg = getattr(model.config, "vision_config", None)
    patch_sz = int(getattr(vcfg, "patch_size", 14) or 14)
    img_sz = int(getattr(vcfg, "image_size", 384) or 384)
    strategy = str(getattr(model.config, "vision_feature_select_strategy", "full"))
    expected_img_tokens = (img_sz // patch_sz) ** 2 + (0 if strategy == "full" else -1 if strategy == "default" else 0)

    train_ex = prepare_scienceqa_batches(
        args.train_data, processor, max_samples=args.max_train_samples, image_token_id=img_tok_id, expected_img_tokens=expected_img_tokens
    )
    val_ex = prepare_scienceqa_batches(
        args.val_data, processor, max_samples=args.max_val_samples, image_token_id=img_tok_id, expected_img_tokens=expected_img_tokens
    )

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    base_acc, base_ppl = evaluate_vlm_scienceqa(model, val_ex, processor, device, batch_size=args.batch_size, max_samples=len(val_ex))
    logger.info("Pre-FT VLM Teacher Baseline: ScienceQA Acc=%.2f%% | Answer PPL=%.2f (n_val=%d)", base_acc, base_ppl, len(val_ex))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_opt_steps = max(1, (len(train_ex) // max(1, args.batch_size * args.grad_accum)) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = base_acc
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_ex)).tolist()
        running_loss = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)

        for step_idx, i in enumerate(range(0, len(perm), args.batch_size), 1):
            b_items = [train_ex[j] for j in perm[i : i + args.batch_size]]
            batch = _collate_vlm_batch(b_items, pad_id=pad_id, eval_prompt_only=False)
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)
            px = batch["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = model(input_ids=in_ids, attention_mask=att_m, pixel_values=px, labels=lbls, use_cache=False)
                loss = out.loss / args.grad_accum

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                running_loss += float(out.loss.item())
                n_batches += 1

            if step_idx % args.grad_accum == 0 or (i + args.batch_size) >= len(perm):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        val_acc, val_ppl = evaluate_vlm_scienceqa(model, val_ex, processor, device, batch_size=args.batch_size, max_samples=len(val_ex))
        logger.info(
            "Epoch %d/%d (%.1fs): avg_loss=%.4f | Val ScienceQA Acc=%.2f%% | Answer PPL=%.2f",
            epoch,
            args.epochs,
            time.time() - t0,
            running_loss / max(1, n_batches),
            val_acc,
            val_ppl,
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            save_copy = model
            save_copy.half().save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            # Restore FP32 for continued fine-tuning on remaining epochs
            if epoch < args.epochs:
                if getattr(model, "multi_modal_projector", None) is not None:
                    model.multi_modal_projector.float()
                model.language_model.float()

    logger.info("Finished ScienceQA Teacher Fine-Tuning: %.2f%% -> Best %.2f%% saved at %s", base_acc, best_acc, out_dir)
    del model, opt, sched, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
