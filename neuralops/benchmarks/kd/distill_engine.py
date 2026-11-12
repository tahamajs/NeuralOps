#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shared Two-Stage Distillation Engine for NeuralOps Benchmarks.

Provides:
  - 2-Stage Distillation:
      * Stage 1: Frozen-Backbone Subspace Hidden Alignment (MSE + Cosine Similarity)
      * Stage 2: Representation & Logit Distillation (Hidden Matching + Top-K Sparse KD / CE)
  - Automatic OOM retry handling (batch-size backoff, gradient checkpointing)
  - Micro-chunked perplexity evaluation avoiding high-vocab VRAM spikes
  - Optimizer memory isolation between distillation stages
"""
from __future__ import annotations

import contextlib
import gc
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuralops.benchmarks.kd.teacher_cache import (
    compute_topk_sparse_kd_loss_from_cache,
    enable_student_gradient_checkpointing,
)

logger = logging.getLogger("neuralops.benchmarks.kd.distill_engine")


def fast_gram_covariance_loss(
    s_hidden: torch.Tensor,
    t_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes normalized Gram matrix covariance alignment across feature dimensions."""
    if attention_mask is not None and attention_mask.dim() == 2:
        mask_bool = attention_mask.bool()
        if mask_bool.any():
            s_flat = s_hidden[mask_bool].float()
            t_flat = t_hidden[mask_bool].float()
        else:
            s_flat = s_hidden.view(-1, s_hidden.size(-1)).float()
            t_flat = t_hidden.view(-1, t_hidden.size(-1)).float()
    else:
        s_flat = s_hidden.view(-1, s_hidden.size(-1)).float()
        t_flat = t_hidden.view(-1, t_hidden.size(-1)).float()

    s_norm = s_flat - s_flat.mean(dim=0, keepdim=True)
    t_norm = t_flat - t_flat.mean(dim=0, keepdim=True)
    g_s = torch.mm(s_norm.t(), s_norm) / max(s_flat.size(0) - 1, 1)
    g_t = torch.mm(t_norm.t(), t_norm) / max(t_flat.size(0) - 1, 1)
    return F.mse_loss(g_s, g_t)


def _forward_hidden(mod: nn.Module, in_ids: torch.Tensor, att_m: Optional[torch.Tensor] = None) -> Any:
    """Invokes module forward pass requesting hidden states while cleanly handling use_cache and masks."""
    kwargs: Dict[str, Any] = {"output_hidden_states": True}
    if att_m is not None:
        kwargs["attention_mask"] = att_m
    try:
        return mod(input_ids=in_ids, use_cache=False, **kwargs)
    except TypeError:
        return mod(input_ids=in_ids, **kwargs)


def _make_grad_scaler(device: torch.device, enabled: bool = True, init_scale: float = 1024.0) -> Any:
    """Create a GradScaler compatible across PyTorch 1.12, 2.0, and 2.6+ without deprecation warnings."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            dev_str = device.type if hasattr(device, "type") else str(device)
            return torch.amp.GradScaler(dev_str if dev_str.startswith("cuda") else "cuda", enabled=enabled, init_scale=init_scale)
        except (TypeError, ValueError):
            pass
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        try:
            return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale)
        except (TypeError, ValueError):
            pass
    try:
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled, init_scale=init_scale)
    except Exception:
        from torch.amp import GradScaler
        return GradScaler("cuda", enabled=enabled, init_scale=init_scale)


def _compute_masked_representation_loss(
    s_h: List[torch.Tensor],
    t_h: List[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute token-masked MSE and cosine distance across layers, excluding pad tokens."""
    if attention_mask is not None and attention_mask.dim() == 2:
        mask_f = attention_mask.float()
        mse_list = []
        cos_list = []
        for s, t in zip(s_h, t_h):
            s_f = s.float()
            t_f = t.float()
            min_l = min(s_f.size(1), t_f.size(1), mask_f.size(1))
            cur_mask = mask_f[:, :min_l]
            cur_s = s_f[:, :min_l, :]
            cur_t = t_f[:, :min_l, :]
            cur_sum = cur_mask.sum().clamp(min=1.0)
            cur_3d = cur_mask.unsqueeze(-1)

            diff_sq = ((cur_s - cur_t) ** 2) * cur_3d
            mse_val = diff_sq.sum() / (cur_sum * cur_s.size(-1))
            mse_list.append(mse_val)

            cos_sim = F.cosine_similarity(cur_s, cur_t, dim=-1)
            cos_dist = ((1.0 - cos_sim) * cur_mask).sum() / cur_sum
            cos_list.append(cos_dist)

        loss_mse = torch.stack(mse_list).mean()
        loss_cos = torch.stack(cos_list).mean()
    else:
        loss_mse = torch.stack([F.mse_loss(s, t) for s, t in zip(s_h, t_h)]).mean()
        loss_cos = torch.stack([1.0 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(s_h, t_h)]).mean()
    return loss_mse, loss_cos


def compute_topk_sparse_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 64,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Computes Top-K Sparse Knowledge Distillation loss between live logits."""
    tau = float(temperature)
    eff_k = min(int(topk), int(teacher_logits.size(-1)))
    t_vals, t_idx = torch.topk(teacher_logits.float(), k=eff_k, dim=-1)
    s_vals = torch.gather(student_logits.float(), dim=-1, index=t_idx)
    p_s = F.log_softmax(s_vals / tau, dim=-1)
    p_t = F.softmax(t_vals / tau, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (tau * tau)


def evaluate_causal_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 15,
) -> float:
    """Computes Causal LM Perplexity with micro-chunked evaluation to prevent high-vocab OOM."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            in_ids = batch["input_ids"].to(device, non_blocking=True)
            att_m = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch.get("labels", in_ids).to(device, non_blocking=True)

            with (torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()):
                lm_sub = getattr(model, "model", getattr(model, "language_model", model))
                if hasattr(lm_sub, "model") and hasattr(lm_sub.model, "layers"):
                    lm_core = lm_sub.model
                else:
                    lm_core = lm_sub

                out = lm_core(input_ids=in_ids, attention_mask=att_m, use_cache=False)
                h = getattr(out, "last_hidden_state", None)
                if h is None and hasattr(out, "hidden_states") and out.hidden_states:
                    h = out.hidden_states[-1]
                if h is None and isinstance(out, tuple):
                    h = out[0]

                if h is None:
                    # Fallback to full forward pass if core model didn't return hidden state
                    full_out = model(input_ids=in_ids, attention_mask=att_m, use_cache=False)
                    logits = full_out.logits
                    seq_len = in_ids.size(1) - 1
                    shift_logits = logits[:, :seq_len, :].contiguous()
                    shift_labels = lbls[:, 1:].contiguous()
                    loss_full = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
                    if torch.isfinite(loss_full):
                        total_loss += loss_full.item()
                        total_tokens += shift_labels.numel()
                    continue

                seq_len = in_ids.size(1) - 1
                targets = lbls[:, 1:].contiguous()
                h_causal = h[:, :seq_len, :]

                chunk_sz = 64
                lm_head = getattr(model, "lm_head", None)
                if lm_head is None and hasattr(model, "language_model"):
                    lm_head = getattr(model.language_model, "lm_head", None)

                for c in range(0, seq_len, chunk_sz):
                    h_c = h_causal[:, c : c + chunk_sz, :]
                    tgt_c = targets[:, c : c + chunk_sz]
                    if lm_head is not None:
                        logits_c = lm_head(h_c)
                        loss_c = F.cross_entropy(logits_c.reshape(-1, logits_c.size(-1)), tgt_c.reshape(-1), reduction="sum")
                        if torch.isfinite(loss_c):
                            total_loss += loss_c.item()
                            total_tokens += tgt_c.numel()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 20.0))


def evaluate_sst2_accuracy(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_samples: int = 200,
) -> float:
    """Evaluates binary sentiment classification on SST-2 validation split."""
    try:
        from datasets import load_dataset
        sst2 = load_dataset("glue", "sst2", split="validation")
    except Exception as exc:
        logger.debug("SST-2 evaluation skipped: %s", exc)
        return 0.0

    model.eval()
    correct = 0
    total = 0

    try:
        pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
        neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
    except Exception:
        pos_id, neg_id = 1, 0

    with torch.no_grad():
        for i, item in enumerate(sst2):
            if i >= max_samples:
                break
            prompt = f"Sentence: {item['sentence']}\nSentiment:"
            enc = tokenizer(prompt, return_tensors="pt")
            in_ids = enc["input_ids"].to(device)
            att_m = enc["attention_mask"].to(device) if "attention_mask" in enc else None

            with (torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()):
                fwd_kwargs: dict[str, Any] = {}
                if att_m is not None:
                    fwd_kwargs["attention_mask"] = att_m
                try:
                    out = model(input_ids=in_ids, use_cache=False, **fwd_kwargs)
                except TypeError:
                    out = model(input_ids=in_ids, **fwd_kwargs)
                last_logits = out.logits[:, -1, :]
                pred_label = 1 if last_logits[0, pos_id] > last_logits[0, neg_id] else 0

            if pred_label == item["label"]:
                correct += 1
            total += 1

    return (correct / max(total, 1)) * 100.0


class TwoStageDistillationEngine:
    """
    Unified 2-Stage Distillation Execution Engine.

    Stage 1:
      Frozen-backbone subspace hidden alignment. Fast gradient-accumulation loop
      matching student hidden states to strided teacher hidden representations.

    Stage 2:
      Fine-tuning representation and logit distillation (Top-K Sparse KD or CE).
    """

    def __init__(
        self,
        train_secs: float = 300.0,
        stage1_frac: float = 0.30,
        stage2_frac: float = 0.70,
        lr_s1: float = 2.0e-4,
        lr_s2: float = 6.0e-5,
        min_lr_s2: float = 1.0e-6,
        weight_decay: float = 0.01,
        grad_accum: int = 4,
        max_grad_norm: float = 1.0,
        topk: int = 64,
        device: Optional[torch.device] = None,
        use_cosine_scheduler: bool = True,
        margin_loss_weight: float = 0.5,
    ) -> None:
        self.train_secs = float(train_secs)
        self.stage1_frac = float(stage1_frac)
        self.stage2_frac = float(stage2_frac)
        self.lr_s1 = float(lr_s1)
        self.lr_s2 = float(lr_s2)
        self.min_lr_s2 = float(min_lr_s2)
        self.weight_decay = float(weight_decay)
        self.grad_accum = max(1, int(grad_accum))
        self.max_grad_norm = float(max_grad_norm)
        self.topk = int(topk)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cosine_scheduler = bool(use_cosine_scheduler)
        self.margin_loss_weight = float(margin_loss_weight)

    def run(
        self,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_model: Optional[nn.Module] = None,
        teacher_cache: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Callable[[nn.Module, DataLoader, torch.device], float]] = None,
        log_prefix: str = "",
        gradient_checkpointing: bool = False,
        train_secs: Optional[float] = None,
        letter_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Execute the two-stage distillation procedure."""
        if teacher_cache is None and teacher_model is None:
            raise ValueError("Either teacher_cache or teacher_model must be provided.")

        dev = self.device
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        if gradient_checkpointing:
            enable_student_gradient_checkpointing(student, log_prefix=log_prefix)

        evaluator = eval_fn or evaluate_causal_ppl
        pre_metric = evaluator(student, val_loader, dev)
        logger.info("%sPre-Distillation Metric: %.2f", log_prefix, pre_metric)

        total_budget_secs = float(train_secs) if train_secs is not None else self.train_secs
        t_start = time.time()
        t_s1_end = t_start + (total_budget_secs * self.stage1_frac)
        t_s2_end = t_start + total_budget_secs

        layer_indices: Optional[List[int]] = teacher_cache.get("layer_indices") if teacher_cache is not None else None

        trainable_params = [p for p in student.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("%sNo trainable parameters found on student model; enabling all params.", log_prefix)
            for p in student.parameters():
                p.requires_grad = True
            trainable_params = list(student.parameters())

        # Stage 1
        opt1 = torch.optim.AdamW(trainable_params, lr=self.lr_s1, weight_decay=self.weight_decay)
        scaler1 = _make_grad_scaler(dev, enabled=(dev.type == "cuda"), init_scale=1024.0)
        student.train()
        step_s1 = 0
        t0_s1 = time.time()

        while time.time() < t_s1_end:
            for batch in train_loader:
                if time.time() >= t_s1_end:
                    break
                in_ids = batch["input_ids"].to(dev, non_blocking=True)
                att_m = batch["attention_mask"].to(dev, non_blocking=True)

                with (torch.cuda.amp.autocast(enabled=(dev.type == "cuda"), dtype=torch.float16) if dev.type == "cuda" else contextlib.nullcontext()):
                    core = getattr(student, "model", getattr(student, "language_model", student))
                    target_s = core.model if (hasattr(core, "model") and hasattr(core.model, "layers")) else core
                    s_out = _forward_hidden(target_s, in_ids, att_m)

                    if "teacher_hidden_states" in batch and layer_indices is not None:
                        t_h_b = batch["teacher_hidden_states"].to(dev, non_blocking=True)
                        s_h = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                        t_h = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                    else:
                        assert teacher_model is not None
                        with torch.no_grad():
                            t_core = getattr(teacher_model, "model", getattr(teacher_model, "language_model", teacher_model))
                            target_t = t_core.model if (hasattr(t_core, "model") and hasattr(t_core.model, "layers")) else t_core
                            t_out = _forward_hidden(target_t, in_ids, att_m)
                        s_h = s_out.hidden_states[1:]
                        t_h = t_out.hidden_states[1:]

                    loss_mse, loss_cos = _compute_masked_representation_loss(s_h, t_h, att_m)
                    loss = (loss_mse + 0.25 * loss_cos) / self.grad_accum

                scaler1.scale(loss).backward()
                step_s1 += 1

                if step_s1 % self.grad_accum == 0:
                    scaler1.unscale_(opt1)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.max_grad_norm)
                    scaler1.step(opt1)
                    scaler1.update()
                    opt1.zero_grad(set_to_none=True)

        if step_s1 > 0 and step_s1 % self.grad_accum != 0:
            scaler1.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(student.parameters(), self.max_grad_norm)
            scaler1.step(opt1)
            scaler1.update()
            opt1.zero_grad(set_to_none=True)

        dur_s1 = time.time() - t0_s1
        del opt1, scaler1
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Stage 2
        opt2 = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=self.lr_s2,
            weight_decay=self.weight_decay,
        )
        scaler2 = _make_grad_scaler(dev, enabled=(dev.type == "cuda"), init_scale=1024.0)
        student.train()
        step_s2 = 0
        t0_s2 = time.time()
        dur_s2_total = max(1e-5, t_s2_end - t0_s2)

        while time.time() < t_s2_end:
            for batch in train_loader:
                if time.time() >= t_s2_end:
                    break
                if self.use_cosine_scheduler:
                    progress = min(1.0, max(0.0, (time.time() - t0_s2) / dur_s2_total))
                    cur_lr = self.min_lr_s2 + 0.5 * (self.lr_s2 - self.min_lr_s2) * (1.0 + math.cos(math.pi * progress))
                    for pg in opt2.param_groups:
                        pg["lr"] = cur_lr

                in_ids = batch["input_ids"].to(dev, non_blocking=True)
                att_m = batch["attention_mask"].to(dev, non_blocking=True)

                with (torch.cuda.amp.autocast(enabled=(dev.type == "cuda"), dtype=torch.float16) if dev.type == "cuda" else contextlib.nullcontext()):
                    core = getattr(student, "model", getattr(student, "language_model", student))
                    target_s = core.model if (hasattr(core, "model") and hasattr(core.model, "layers")) else core
                    s_out = _forward_hidden(target_s, in_ids, att_m)

                    if "teacher_hidden_states" in batch and layer_indices is not None:
                        t_h_b = batch["teacher_hidden_states"].to(dev, non_blocking=True)
                        s_h = [s_out.hidden_states[idx + 1] for idx in layer_indices]
                        t_h = [t_h_b[:, j, :, :] for j in range(len(layer_indices))]
                    else:
                        assert teacher_model is not None
                        with torch.no_grad():
                            t_core = getattr(teacher_model, "model", getattr(teacher_model, "language_model", teacher_model))
                            target_t = t_core.model if (hasattr(t_core, "model") and hasattr(t_core.model, "layers")) else t_core
                            t_out = _forward_hidden(target_t, in_ids, att_m)
                        s_h = s_out.hidden_states[1:]
                        t_h = t_out.hidden_states[1:]

                    loss_mse, loss_cos = _compute_masked_representation_loss(s_h, t_h, att_m)

                    # Compute logit loss if available
                    lm_head = getattr(student, "lm_head", None)
                    if lm_head is None and hasattr(student, "language_model"):
                        lm_head = getattr(student.language_model, "lm_head", None)

                    # Compute auxiliary multiple-choice option loss & margin preservation if metadata present
                    loss_choice = torch.tensor(0.0, device=dev)
                    s_last = s_out.hidden_states[-1] if hasattr(s_out, "hidden_states") and s_out.hidden_states else None
                    if lm_head is not None and s_last is not None and "answer_idx" in batch and "prompt_lens" in batch and self.margin_loss_weight > 0:
                        ans_indices = batch["answer_idx"].to(dev, non_blocking=True)
                        prompt_lens = batch["prompt_lens"].to(dev, non_blocking=True)
                        num_choices_t = batch.get("num_choices", None)
                        c_losses = []
                        cand_ids = letter_token_ids
                        for b_i in range(in_ids.size(0)):
                            pos = int(prompt_lens[b_i].item()) - 1
                            tgt = ans_indices[b_i]
                            n_ch = int(num_choices_t[b_i].item()) if num_choices_t is not None else (len(cand_ids) if cand_ids else 0)
                            if cand_ids is not None and 0 <= pos < s_last.size(1) and 0 <= tgt.item() < n_ch:
                                sub_ids = [cid for cid in cand_ids[:n_ch] if 0 <= cid < lm_head.out_features]
                                if len(sub_ids) == n_ch:
                                    step_hidden = s_last[b_i : b_i + 1, pos : pos + 1, :]
                                    full_logits = lm_head(step_hidden)
                                    opt_logits = full_logits[0, 0, sub_ids].unsqueeze(0)
                                    ce_c = F.cross_entropy(opt_logits, tgt.unsqueeze(0))
                                    # Margin preservation: enforce positive logit gap to closest competitor
                                    correct_val = opt_logits[0, tgt]
                                    mask = torch.ones(n_ch, dtype=torch.bool, device=dev)
                                    mask[tgt] = False
                                    if mask.any():
                                        max_other = opt_logits[0, mask].max()
                                        margin_penalty = F.relu(1.0 - (correct_val - max_other))
                                        ce_c = ce_c + 0.20 * margin_penalty
                                    if torch.isfinite(ce_c):
                                        c_losses.append(ce_c)
                        if c_losses:
                            loss_choice = torch.stack(c_losses).mean()

                    if lm_head is not None and "teacher_topk_vals" in batch and "teacher_topk_idx" in batch:
                        s_logits = lm_head(s_last)
                        loss_kd = compute_topk_sparse_kd_loss_from_cache(
                            s_logits,
                            batch["teacher_topk_vals"].to(dev, non_blocking=True),
                            batch["teacher_topk_idx"].to(dev, non_blocking=True),
                        )
                        loss = (loss_mse + 0.50 * loss_cos + 0.25 * loss_kd + self.margin_loss_weight * loss_choice) / self.grad_accum
                    elif lm_head is not None and s_last is not None:
                        logits_shift = lm_head(s_last[:, :-1, :]).contiguous()
                        labels_shift = in_ids[:, 1:].contiguous()
                        ce_loss = F.cross_entropy(logits_shift.view(-1, logits_shift.size(-1)), labels_shift.view(-1))
                        ce_term = ce_loss if torch.isfinite(ce_loss) else 0.0
                        loss = (loss_mse + 0.50 * loss_cos + 0.25 * ce_term + self.margin_loss_weight * loss_choice) / self.grad_accum
                    else:
                        loss = (loss_mse + 0.50 * loss_cos + self.margin_loss_weight * loss_choice) / self.grad_accum

                scaler2.scale(loss).backward()
                step_s2 += 1

                if step_s2 % self.grad_accum == 0:
                    scaler2.unscale_(opt2)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.max_grad_norm)
                    scaler2.step(opt2)
                    scaler2.update()
                    opt2.zero_grad(set_to_none=True)
                    if self.use_cosine_scheduler:
                        progress = min(1.0, max(0.0, (time.time() - t0_s2) / dur_s2_total))
                        cur_lr = self.min_lr_s2 + 0.5 * (self.lr_s2 - self.min_lr_s2) * (1.0 + math.cos(math.pi * progress))
                        for pg in opt2.param_groups:
                            pg["lr"] = cur_lr

        if step_s2 > 0 and step_s2 % self.grad_accum != 0:
            scaler2.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(student.parameters(), self.max_grad_norm)
            scaler2.step(opt2)
            scaler2.update()
            opt2.zero_grad(set_to_none=True)
            if self.use_cosine_scheduler:
                progress = min(1.0, max(0.0, (time.time() - t0_s2) / dur_s2_total))
                cur_lr = self.min_lr_s2 + 0.5 * (self.lr_s2 - self.min_lr_s2) * (1.0 + math.cos(math.pi * progress))
                for pg in opt2.param_groups:
                    pg["lr"] = cur_lr

        dur_s2 = time.time() - t0_s2
        if self.use_cosine_scheduler:
            final_progress = min(1.0, max(0.0, dur_s2 / dur_s2_total))
            final_lr_s2 = self.min_lr_s2 + 0.5 * (self.lr_s2 - self.min_lr_s2) * (1.0 + math.cos(math.pi * final_progress))
        else:
            final_lr_s2 = float(opt2.param_groups[0]["lr"])
        del opt2, scaler2
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        total_time = time.time() - t_start
        post_metric = evaluator(student, val_loader, dev)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024)) if dev.type == "cuda" else 0

        logger.info(
            "%sDistillation Finished: pre=%.2f, post=%.2f, delta=%+.2f, steps=%d+%d, dur=%.1fs, peak_vram=%d MB",
            log_prefix,
            pre_metric,
            post_metric,
            post_metric - pre_metric,
            step_s1,
            step_s2,
            total_time,
            peak_vram,
        )

        return {
            "pre_metric": float(pre_metric),
            "post_metric": float(post_metric),
            "delta": float(post_metric - pre_metric),
            "train_secs": float(total_time),
            "dur_s1": float(dur_s1),
            "dur_s2": float(dur_s2),
            "steps_s1": int(step_s1),
            "steps_s2": int(step_s2),
            "total_steps": int(step_s1 + step_s2),
            "peak_vram_mb": int(peak_vram),
            "final_lr_s2": float(final_lr_s2),
        }


def execute_with_oom_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_batch_size: int = 8,
    on_oom_hook: Optional[Callable[[int], None]] = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with automatic batch-size halving and VRAM cleanup on CUDA OOM."""
    cur_bs = initial_batch_size
    for attempt in range(max_retries):
        try:
            return fn(*args, batch_size=cur_bs, **kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                logger.warning("[OOM RETRY] CUDA out of memory; backing off to batch_size=%d (attempt %d/%d)", cur_bs, attempt + 1, max_retries)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.05)
                if on_oom_hook is not None:
                    on_oom_hook(cur_bs)
                continue
            raise


__all__ = [
    "fast_gram_covariance_loss",
    "compute_topk_sparse_kd_loss",
    "evaluate_causal_ppl",
    "evaluate_sst2_accuracy",
    "TwoStageDistillationEngine",
    "execute_with_oom_retry",
]
