from typing import Any
import json
import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm


from framework.core.memory import cleanup_gpu_memory, get_gpu_memory_stats


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config,
    num_labels: int,
    is_multichoice: bool = False,
) -> Dict[str, float]:
    """Train model with maximum GPU compute utilization and robust memory cleanup."""
    cleanup_gpu_memory(force=True)
    model.to(device)
    model.train()

    use_amp = getattr(config, "fp16", False) or getattr(config, "amp", True)
    amp_dtype = torch.float16 if device.type == "cuda" else (torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # Optimizer with fused option if available on CUDA
    optimizer_kwargs = {"lr": getattr(config, "learning_rate", 2e-5), "weight_decay": getattr(config, "weight_decay", 0.01)}
    if device.type == "cuda" and hasattr(torch.optim.AdamW, "fused"):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    epochs = getattr(config, "epochs", 3)
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    steps_per_epoch = max(1, len(train_loader))
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type in ["cuda", "mps"]):
                if is_multichoice:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    true_labels = batch["labels"].to(device, non_blocking=True)
                    bs, nc, sl = input_ids.shape
                    flat_ids = input_ids.view(-1, sl)
                    flat_mask = attention_mask.view(-1, sl)
                    outputs = model(input_ids=flat_ids, attention_mask=flat_mask)
                    logits = outputs["logits"]
                    combined_labels = true_labels.repeat_interleave(nc)
                    loss = nn.functional.cross_entropy(logits, combined_labels)
                else:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    loss = nn.functional.cross_entropy(logits, labels)

            if torch.isfinite(loss):
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

                curr_loss = loss.detach().item()
                epoch_loss += curr_loss
                total_loss += curr_loss
            else:
                curr_loss = 0.0

            if (step + 1) % max(1, getattr(config, "logging_steps", 10)) == 0 or step == len(train_loader) - 1:
                pbar.set_postfix({"loss": f"{curr_loss:.4f}"})

            # Explicitly delete activation references to free memory immediately
            del outputs, logits, loss

        # End of epoch memory cleanup
        cleanup_gpu_memory(force=False)

    val_acc = evaluate_model(model, val_loader, device, num_labels, is_multichoice)
    cleanup_gpu_memory(force=True)
    return {"final_accuracy": val_acc, "avg_loss": total_loss / (epochs * steps_per_epoch)}


def evaluate_model(
    model,
    data_loader,
    device,
    num_labels,
    is_multichoice=False,
    compute_all_metrics=False,
) -> Union[float, Dict[str, float]]:
    """Evaluate model with non-blocking GPU execution and memory safety."""
    cleanup_gpu_memory(force=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if is_multichoice:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                true_labels = batch["labels"]
                bs, nc, sl = input_ids.shape
                flat_ids = input_ids.view(-1, sl)
                flat_mask = attention_mask.view(-1, sl)
                outputs = model(input_ids=flat_ids, attention_mask=flat_mask)
                logits = outputs["logits"]
                avg_logits = logits.view(bs, nc, -1).mean(dim=1)
                preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(true_labels.numpy())
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())
            del outputs

    cleanup_gpu_memory(force=False)
    if not all_preds:
        return 0.0 if not compute_all_metrics else {"accuracy": 0.0}
    acc = accuracy_score(all_labels, all_preds)
    if not compute_all_metrics:
        return acc
    return {
        "accuracy": acc,
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "matthews_corrcoef": matthews_corrcoef(all_labels, all_preds),
    }


def benchmark_model(model, data_loader, device, warmup=2) -> Dict[str, float]:
    """Benchmark model."""
    model.eval()
    for i, batch in enumerate(data_loader):
        if i >= warmup:
            break
        if batch["input_ids"].dim() == 3:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            for j in range(input_ids.shape[1]):
                _ = model(input_ids=input_ids[:, j, :], attention_mask=attn[:, j, :])
        else:
            _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time.time()
    batch_times = []
    with torch.no_grad():
        for batch in data_loader:
            b_start = time.time()
            if batch["input_ids"].dim() == 3:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                for j in range(input_ids.shape[1]):
                    _ = model(
                        input_ids=input_ids[:, j, :], attention_mask=attn[:, j, :]
                    )
            else:
                _ = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.time() - b_start)
    total_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    return {
        "total_time": total_time,
        "avg_batch_time": np.mean(batch_times) if batch_times else 0,
        "std_batch_time": np.std(batch_times) if batch_times else 0,
        "samples_per_second": (
            len(data_loader.dataset) / total_time if total_time > 0 else 0
        ),
        "memory_usage_mb": memory_after - memory_before,
        "peak_memory_mb": memory_after,
        "throughput": len(data_loader.dataset) / total_time if total_time > 0 else 0,
    }


def count_parameters(model, trainable_only=True) -> int:
    """Count parameters."""
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_only
        else sum(p.numel() for p in model.parameters())
    )


def compute_metrics(eval_pred, task: str = "classification") -> Any:
    """Compute and return metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {"accuracy": accuracy_score(labels, preds)}
    if task == "classification":
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
    if len(np.unique(labels)) == 2:
        metrics["matthews_corrcoef"] = matthews_corrcoef(labels, preds)
    return metrics


def save_training_history(history: List[Dict[str, Any]], save_path: str) -> None:
    """Persist training history to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: List[Dict[str, Any]], save_path: str) -> None:
    """Plot training curves."""
    import matplotlib.pyplot as plt

    losses = [h.get("loss") for h in history if "loss" in h]
    eval_losses = [h.get("eval_loss") for h in history if "eval_loss" in h]
    eval_acc = [h.get("eval_accuracy") for h in history if "eval_accuracy" in h]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if losses:
        axes[0].plot(range(len(losses)), losses, "b-", label="Train Loss")
    if eval_losses:
        axes[0].plot(range(len(eval_losses)), eval_losses, "r-", label="Val Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if eval_acc:
        axes[1].plot(
            range(len(eval_acc)), eval_acc, "g-", marker="o", label="Val Accuracy"
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
