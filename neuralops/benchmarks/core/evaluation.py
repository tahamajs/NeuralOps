"""Evaluation metrics computation and hardware profiling for benchmarks."""
from __future__ import annotations

import logging
import math
import time
from typing import Any
import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)


def evaluate_classification(
    model: nn.Module,
    eval_loader: Any,
    device: str | torch.device = "cpu",
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate classification model and return accuracy, f1, precision, recall, and loss.
    """
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    import sys
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(
        eval_loader,
        desc="Evaluating",
        file=sys.stderr,
        leave=False,
    ) if has_tqdm else eval_loader

    with torch.no_grad():
        for b_idx, batch in enumerate(pbar):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            total_loss += float(loss.item()) if loss is not None else 0.0
            num_batches += 1

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if max_eval_batches is not None and b_idx + 1 >= max_eval_batches:
                break

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(num_batches, 1)

    # Simple f1 calculation for binary classification
    tp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "loss": float(avg_loss),
    }


def evaluate_perplexity(
    model: nn.Module,
    eval_loader: Any,
    device: str | torch.device = "cpu",
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate causal language model perplexity on sequence chunks."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(eval_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            # Causal LM standard target
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                total_loss += float(loss.item()) * input_ids.size(0)
                total_tokens += input_ids.numel()
                num_batches += 1

            if max_eval_batches is not None and b_idx + 1 >= max_eval_batches:
                break

    avg_loss = total_loss / max(num_batches * eval_loader.batch_size, 1)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return {
        "perplexity": float(ppl),
        "loss": float(avg_loss),
    }


def measure_resources(
    model: nn.Module,
    sample_batch: tuple[torch.Tensor, ...],
    device: str | torch.device = "cpu",
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> dict[str, float]:
    """
    Profile peak VRAM consumption, latency per token, and token throughput.
    """
    model.eval()
    input_ids = sample_batch[0].to(device)
    attention_mask = sample_batch[1].to(device)
    batch_size, seq_len = input_ids.shape
    total_tokens_per_pass = batch_size * seq_len

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = peak_bytes / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time = time.perf_counter() - start_time
    total_tokens = total_tokens_per_pass * measure_iters

    throughput = total_tokens / max(total_time, 1e-8)
    latency_ms_per_token = (total_time / max(total_tokens, 1)) * 1000.0

    return {
        "peak_vram_mb": float(peak_vram_mb),
        "latency_ms_per_token": float(latency_ms_per_token),
        "throughput_tok_per_sec": float(throughput),
    }


def measure_peak_memory_mb(device: str | torch.device = "cpu") -> float:
    """Return peak memory allocated in MB."""
    if torch.cuda.is_available() and device != "cpu":
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return 0.0


def measure_latency_and_throughput(
    model: nn.Module,
    dataloader: Any,
    device: str | torch.device = "cpu",
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> dict[str, float]:
    """Measure latency and token throughput across sample batches."""
    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    if sample_batch is None:
        return {"latency_ms_per_token": 0.0, "throughput_tok_per_sec": 0.0}

    stats = measure_resources(
        model=model,
        sample_batch=sample_batch,
        device=device,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    return {
        "latency_ms_per_token": stats["latency_ms_per_token"],
        "throughput_tok_per_sec": stats["throughput_tok_per_sec"],
    }


__all__ = [
    "evaluate_classification",
    "evaluate_perplexity",
    "measure_resources",
    "measure_latency_and_throughput",
    "measure_peak_memory_mb",
]

