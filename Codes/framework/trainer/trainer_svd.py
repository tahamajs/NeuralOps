import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config,
    num_labels: int,
    is_multichoice: bool = False,
) -> Any:
    """Train model."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps_per_epoch = len(train_loader)
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            if is_multichoice:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                true_labels = batch["labels"].to(device)
                batch_size, num_choices, seq_len = input_ids.shape
                input_ids_flat = input_ids.view(-1, seq_len)
                attention_mask_flat = attention_mask.view(-1, seq_len)
                outputs = model(
                    input_ids=input_ids_flat, attention_mask=attention_mask_flat
                )
                logits = outputs["logits"]
                combined_labels = true_labels.repeat_interleave(num_choices)
                loss = nn.functional.cross_entropy(logits, combined_labels)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(logits, labels)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    val_accuracy = evaluate_model(
        model, val_loader, device, num_labels, is_multichoice=is_multichoice
    )
    return {"final_accuracy": val_accuracy, "avg_loss": epoch_loss / steps_per_epoch}


def evaluate_model(model, data_loader, device, num_labels, is_multichoice=False) -> Any:
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            if is_multichoice:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                true_labels = batch["labels"]
                batch_size, num_choices, _ = input_ids.shape
                input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
                attention_mask_flat = attention_mask.view(-1, input_ids.shape[-1])
                outputs = model(
                    input_ids=input_ids_flat, attention_mask=attention_mask_flat
                )
                logits = outputs["logits"]
                logits_reshaped = logits.view(batch_size, num_choices, -1)
                avg_logits = logits_reshaped.mean(dim=1)
                preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(true_labels.numpy())
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())
    if len(all_preds) == 0:
        return 0.0
    return accuracy_score(all_labels, all_preds)


def benchmark_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> dict:
    """Benchmark model."""
    model.eval()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time.time()
    batch_times = []
    with torch.no_grad():
        for batch in data_loader:
            b_start = time.time()
            if batch["input_ids"].dim() == 3:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                for i in range(input_ids.shape[1]):
                    _ = model(
                        input_ids=input_ids[:, i, :],
                        attention_mask=attention_mask[:, i, :],
                    )
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.time() - b_start)
    total_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    return {
        "total_time": total_time,
        "avg_batch_time": np.mean(batch_times) if batch_times else 0,
        "samples_per_second": (
            len(data_loader.dataset) / total_time if total_time > 0 else 0
        ),
        "memory_usage_mb": memory_after - memory_before,
        "peak_memory_mb": memory_after,
        "throughput": len(data_loader.dataset) / total_time if total_time > 0 else 0,
    }


def count_parameters(model: nn.Module) -> int:
    """Count parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
