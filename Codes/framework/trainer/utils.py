# framework/training/utils.py
"""
Training utilities for model training, evaluation, and benchmarking.
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from typing import Dict, Optional, Any, List
import json
from pathlib import Path


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Any,
    num_labels: int,
    is_multichoice: bool = False
) -> Dict[str, float]:
    """
    Train a model with custom training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        config: Configuration with 'learning_rate', 'epochs'
        num_labels: Number of labels
        is_multichoice: Whether task is multiple choice
        
    Returns:
        Dictionary with training results
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps_per_epoch = len(train_loader)
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            
            if is_multichoice:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                true_labels = batch['labels'].to(device)
                batch_size, num_choices, seq_len = input_ids.shape
                
                input_ids_flat = input_ids.view(-1, seq_len)
                attention_mask_flat = attention_mask.view(-1, seq_len)
                outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                logits = outputs['logits']
                combined_labels = true_labels.repeat_interleave(num_choices)
                loss = nn.functional.cross_entropy(logits, combined_labels)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                loss = nn.functional.cross_entropy(logits, labels)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_accuracy = evaluate_model(model, val_loader, device, num_labels, is_multichoice)
    
    return {
        'final_accuracy': val_accuracy,
        'avg_loss': epoch_loss / steps_per_epoch
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    is_multichoice: bool = False,
    compute_all_metrics: bool = False
) -> Dict[str, float] or float:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to evaluate on
        num_labels: Number of labels
        is_multichoice: Whether task is multiple choice
        compute_all_metrics: Whether to compute all metrics
        
    Returns:
        Dictionary of metrics or just accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if is_multichoice:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                true_labels = batch['labels']
                batch_size, num_choices, _ = input_ids.shape
                
                input_ids_flat = input_ids.view(-1, input_ids.shape[-1])
                attention_mask_flat = attention_mask.view(-1, input_ids.shape[-1])
                outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                logits = outputs['logits']
                logits_reshaped = logits.view(batch_size, num_choices, -1)
                avg_logits = logits_reshaped.mean(dim=1)
                preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(true_labels.numpy())
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())
    
    if len(all_preds) == 0:
        return 0.0 if not compute_all_metrics else {'accuracy': 0.0}
    
    if not compute_all_metrics:
        return accuracy_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(all_labels, all_preds)
    }


def benchmark_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    warmup: int = 2
) -> Dict[str, float]:
    """
    Benchmark model inference speed and memory usage.
    
    Args:
        model: Model to benchmark
        data_loader: Data loader for benchmarking
        device: Device to benchmark on
        warmup: Number of warmup batches
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    # Warmup
    for i, batch in enumerate(data_loader):
        if i >= warmup:
            break
        if batch['input_ids'].dim() == 3:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            for j in range(input_ids.shape[1]):
                _ = model(input_ids=input_ids[:, j, :], attention_mask=attention_mask[:, j, :])
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Actual benchmarking
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time.time()
    batch_times = []
    
    with torch.no_grad():
        for batch in data_loader:
            b_start = time.time()
            
            if batch['input_ids'].dim() == 3:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                for j in range(input_ids.shape[1]):
                    _ = model(input_ids=input_ids[:, j, :], attention_mask=attention_mask[:, j, :])
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_times.append(time.time() - b_start)
    
    total_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'std_batch_time': np.std(batch_times) if batch_times else 0,
        'samples_per_second': len(data_loader.dataset) / total_time if total_time > 0 else 0,
        'memory_usage_mb': memory_after - memory_before,
        'peak_memory_mb': memory_after,
        'throughput': len(data_loader.dataset) / total_time if total_time > 0 else 0
    }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model.
    
    Args:
        model: Model to count parameters for
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_metrics(eval_pred, task: str = "classification"):
    """
    Compute metrics based on task type.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        task: Task type ('classification', 'regression', 'multi_class')
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
    }
    
    if len(np.unique(labels)) == 2:
        metrics['matthews_corrcoef'] = matthews_corrcoef(labels, predictions)
    
    return metrics


def save_training_history(history: List[Dict[str, Any]], save_path: str):
    """
    Save training history to JSON file.
    
    Args:
        history: List of training logs
        save_path: Path to save the file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: List[Dict[str, Any]], save_path: str):
    """
    Plot training curves.
    
    Args:
        history: List of training logs
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    losses = [h.get('loss') for h in history if 'loss' in h]
    eval_losses = [h.get('eval_loss') for h in history if 'eval_loss' in h]
    eval_acc = [h.get('eval_accuracy') for h in history if 'eval_accuracy' in h]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if losses:
        axes[0].plot(range(len(losses)), losses, 'b-', label='Training Loss')
    if eval_losses:
        axes[0].plot(range(len(eval_losses)), eval_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if eval_acc:
        axes[1].plot(range(len(eval_acc)), eval_acc, 'g-', marker='o', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()