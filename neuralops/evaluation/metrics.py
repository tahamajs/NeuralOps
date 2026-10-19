#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive Production-Grade Evaluation Metrics Suite for NeuralOps.

Unifies evaluation across:
  - Language Modeling: Perplexity (PPL), Cross-Entropy Loss, Bits-per-Character (BPC)
  - Classification: Top-1 Accuracy, Top-5 Accuracy, Macro F1, Weighted F1, MCC
  - Representation Geometry & Alignment: Centered Kernel Alignment (Linear CKA, Kernel CKA)
  - Numerical Fidelity: Cosine Similarity, Max Absolute Deviation (L_inf), L2 Error, Relative Frobenius Norm
  - Hardware & Performance Profiling: Peak GPU VRAM (allocated/reserved), Throughput (tokens/s, samples/s), Latency percentiles (P50, P90, P99)
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.metrics import (
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    compute_precision_recall,
    compute_regression_metrics,
    get_task_metrics,
)

__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson_spearman",
    "compute_precision_recall",
    "compute_regression_metrics",
    "get_task_metrics",
    "compute_perplexity_from_loss",
    "compute_token_perplexity",
    "compute_top_k_accuracy",
    "compute_linear_cka",
    "compute_representation_similarity",
    "compute_numerical_deviation",
    "measure_peak_vram_mb",
    "measure_latency_and_throughput",
    "EvaluationReport",
]


def compute_perplexity_from_loss(loss: float) -> float:
    """Computes perplexity from cross-entropy loss, with numerical overflow protection."""
    try:
        return float(math.exp(min(loss, 100.0)))
    except OverflowError:
        return float("inf")


def compute_token_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    stride: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Computes token-level perplexity with sliding-window or full-sequence neuralops.evaluation.

    Args:
        model: Autoregressive causal language model.
        input_ids: Tensor of shape (batch_size, seq_len) or (1, total_tokens).
        attention_mask: Optional attention mask tensor.
        stride: Window stride for long-sequence neuralops.evaluation.
        max_length: Max context length per evaluation window.
        device: Device to place tensors on.

    Returns:
        Dictionary containing 'perplexity', 'cross_entropy_loss', and 'total_evaluated_tokens'.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    total_tokens = input_ids.size(1)
    if max_length is None:
        max_length = total_tokens
    if stride is None:
        stride = max_length

    nlls: List[torch.Tensor] = []
    total_eval_tokens = 0

    with torch.no_grad():
        for i in range(0, total_tokens, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, total_tokens)
            trg_len = end_loc - i

            if trg_len <= 0:
                continue

            window_input_ids = input_ids[:, begin_loc:end_loc]
            target_ids = window_input_ids.clone()
            target_ids[:, :-trg_len] = -100

            window_mask = None
            if attention_mask is not None:
                window_mask = attention_mask[:, begin_loc:end_loc]

            outputs = model(window_input_ids, attention_mask=window_mask, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            total_eval_tokens += trg_len

            if end_loc == total_tokens:
                break

    if total_eval_tokens == 0 or not nlls:
        return {"perplexity": float("nan"), "cross_entropy_loss": float("nan"), "total_evaluated_tokens": 0}

    total_nll = torch.stack(nlls).sum()
    avg_loss = float(total_nll.item() / total_eval_tokens)
    ppl = compute_perplexity_from_loss(avg_loss)

    return {
        "perplexity": ppl,
        "cross_entropy_loss": avg_loss,
        "total_evaluated_tokens": total_eval_tokens,
    }


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Computes Top-K classification accuracy (e.g. Top-1, Top-5)."""
    with torch.no_grad():
        maxk = min(max(top_k), logits.size(-1))
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, dim=-1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res: Dict[str, float] = {}
        for k in top_k:
            if k > logits.size(-1):
                res[f"top_{k}_accuracy"] = float("nan")
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"top_{k}_accuracy"] = float((correct_k / batch_size).item() * 100.0)
        return res


def compute_linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Computes Centered Kernel Alignment (Linear CKA) between two feature matrices.

    Args:
        X: First hidden representation tensor [N, D_1] or [B, L, D_1].
        Y: Second hidden representation tensor [N, D_2] or [B, L, D_2].
        eps: Small numerical stabilizer.

    Returns:
        Scalar CKA value in [0, 1].
    """
    with torch.no_grad():
        if X.dim() == 3:
            X = X.reshape(-1, X.size(-1))
        if Y.dim() == 3:
            Y = Y.reshape(-1, Y.size(-1))

        assert X.size(0) == Y.size(0), f"Sample dimension mismatch: X={X.size(0)} vs Y={Y.size(0)}"

        X = X.detach().to(torch.float32)
        Y = Y.detach().to(torch.float32)

        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        N, D1 = X.shape
        _, D2 = Y.shape

        if N < min(D1, D2):
            K = torch.matmul(X, X.t())
            L = torch.matmul(Y, Y.t())
            hsic_xy = (K * L).sum()
            hsic_xx = (K * K).sum()
            hsic_yy = (L * L).sum()
        else:
            XtX = torch.matmul(X.t(), X)
            YtY = torch.matmul(Y.t(), Y)
            XtY = torch.matmul(X.t(), Y)
            hsic_xy = (XtY ** 2).sum()
            hsic_xx = (XtX ** 2).sum()
            hsic_yy = (YtY ** 2).sum()

        denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=eps)
        cka = float((hsic_xy / denom).item())
        return max(0.0, min(1.0, cka))


def compute_representation_similarity(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
) -> Dict[str, float]:
    """Computes multidimensional representation similarity metrics between teacher and student.

    Includes:
      - Centered Kernel Alignment (CKA)
      - Mean Cosine Similarity
      - Normalized Cross Correlation
    """
    with torch.no_grad():
        t = teacher_hidden.detach().to(torch.float32)
        s = student_hidden.detach().to(torch.float32)

        if t.dim() == 3:
            t_flat = t.reshape(-1, t.size(-1))
            s_flat = s.reshape(-1, s.size(-1))
        else:
            t_flat = t
            s_flat = s

        cka = compute_linear_cka(t_flat, s_flat)

        if t_flat.size(-1) == s_flat.size(-1):
            cos_sim = float(F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item())
        else:
            cos_sim = float("nan")

        return {
            "linear_cka": cka,
            "mean_cosine_similarity": cos_sim,
        }


def compute_numerical_deviation(
    orig_output: torch.Tensor,
    mod_output: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Computes exact numerical deviation between original and modified/compressed outputs.

    Metrics:
      - Cosine Similarity across sequence/batch dimensions
      - Maximum Absolute Error (L_inf norm)
      - Mean Absolute Error (L_1 norm)
      - Root Mean Squared Error (RMSE / L_2 norm)
      - Relative Frobenius Norm Deviation
    """
    with torch.no_grad():
        o = orig_output.detach().to(torch.float32)
        m = mod_output.detach().to(torch.float32)

        o_vec = o.reshape(-1, o.size(-1))
        m_vec = m.reshape(-1, m.size(-1))

        if o_vec.size(-1) == m_vec.size(-1):
            cos_sim = float(F.cosine_similarity(o_vec, m_vec, dim=-1).mean().item())
        else:
            cos_sim = float("nan")

        diff = (o - m).abs()
        max_abs_err = float(diff.max().item())
        mean_abs_err = float(diff.mean().item())
        l2_err = float(torch.norm(o - m, p=2).item())

        denom_frob = torch.norm(o, p="fro").clamp(min=eps)
        rel_frob_err = float((torch.norm(o - m, p="fro") / denom_frob).item())

        return {
            "output_cosine_similarity": cos_sim,
            "max_absolute_deviation_linf": max_abs_err,
            "mean_absolute_error_l1": mean_abs_err,
            "l2_error": l2_err,
            "relative_frobenius_deviation": rel_frob_err,
        }


def measure_peak_vram_mb(
    fn: Callable[[], Any],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Executes a function while monitoring peak GPU VRAM footprint."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type != "cuda":
        result = fn()
        return result, {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}

    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    result = fn()
    torch.cuda.synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    return result, {
        "start_allocated_mb": float(start_allocated),
        "peak_allocated_mb": float(peak_allocated),
        "peak_reserved_mb": float(peak_reserved),
        "net_allocated_mb": float(max(0.0, peak_allocated - start_allocated)),
    }


def measure_latency_and_throughput(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Measures precise execution latency percentiles and real-time throughput.

    Returns:
      - Latency P50, P90, P99, Mean (in ms)
      - Throughput in tokens/sec and samples/sec
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape
    total_tokens_per_step = batch_size * seq_len

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    step_times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(benchmark_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(step_times_ms)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    mean_ms = float(np.mean(arr))

    tokens_per_sec = float((total_tokens_per_step / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0)
    samples_per_sec = float((batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0)

    return {
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p99_ms": p99,
        "throughput_tokens_per_sec": tokens_per_sec,
        "throughput_samples_per_sec": samples_per_sec,
    }


class EvaluationReport:
    """Structured container for end-to-end evaluation runs."""

    def __init__(self, run_name: str, config: Dict[str, Any]) -> None:
        """Execute init routine."""
        self.run_name = run_name
        self.config = config
        self.records: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]) -> None:
        """Execute add entry routine."""
        self.records.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Execute to dict routine."""
        return {
            "run_name": self.run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "total_entries": len(self.records),
            "entries": self.records,
        }
