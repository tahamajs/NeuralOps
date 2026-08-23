"""Inference and metric evaluation runner for test splits."""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from ..core.metrics import compute_metrics
from ..tasks.collator import DataCollatorForGLUE
from ..tasks.glue_tasks import get_task_info
from transformers import AutoTokenizer


def evaluate_model(
    model: nn.Module,
    eval_dataset: Dataset,
    task_name: str,
    batch_size: int = 64,
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Execute forward inference over evaluation split and compute task-specific metric scores.

    Args:
        model: Trained sequence classification model.
        eval_dataset: Evaluation split Dataset.
        task_name: GLUE task name.
        batch_size: Evaluation batch size.
        tokenizer: Optional tokenizer instance for collator.

    Returns:
        Dictionary containing validation metric scores.
    """
    model.eval()
    device = next(model.parameters()).device
    task_info = get_task_info(task_name)
    is_regression = task_info["is_regression"]

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    collator = DataCollatorForGLUE(tokenizer=tokenizer, is_regression=is_regression)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            if is_regression:
                preds = logits.squeeze(-1).cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(task_name, all_preds, all_labels)
    return metrics
