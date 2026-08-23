"""Layer 1: End-to-end GLUE task fine-tuning runner with early collapse and NaN detection."""

from __future__ import annotations
import time
import math
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from datasets import Dataset

from ..core.seed import set_seed
from ..tasks.collator import DataCollatorForGLUE
from ..tasks.glue_tasks import get_task_info
from .evaluate import evaluate_model

logger = logging.getLogger("benchmark.train")


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    task_name: str,
    config: Dict[str, Any],
    seed: int = 42,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a model on a GLUE task and evaluate on the validation split.

    Args:
        model: Transformer model instance to train.
        train_dataset: Pre-tokenized training split.
        eval_dataset: Pre-tokenized validation split.
        task_name: Target task ('sst2', 'mrpc', etc.).
        config: Global benchmark configuration dictionary.
        seed: Random seed for data shuffling and initializations.
        lr: Target learning rate (defaults to config).
        batch_size: Training batch size (defaults to config).
        epochs: Number of training epochs (defaults to task-specific config).

    Returns:
        Detailed training summary dictionary including status, final metrics, and loss history.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    task_info = get_task_info(task_name)
    is_regression = task_info["is_regression"]

    # Resolve training hyperparameters
    lr_val = lr if lr is not None else float(config.get("learning_rates", [3e-5])[0])
    bs_val = batch_size if batch_size is not None else int(config.get("batch_size", 32))
    task_epochs = config.get("epochs", {}).get(task_name.lower(), 3)
    epochs_val = epochs if epochs is not None else task_epochs
    weight_decay = float(config.get("weight_decay", 0.01))
    grad_clip = float(config.get("gradient_clip", 1.0))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))

    tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", "distilbert-base-uncased"))
    collator = DataCollatorForGLUE(tokenizer=tokenizer, is_regression=is_regression)
    train_dataloader = DataLoader(train_dataset, batch_size=bs_val, shuffle=True, collate_fn=collator)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_val)

    total_steps = len(train_dataloader) * epochs_val
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision context
    use_cuda_amp = torch.cuda.is_available() and config.get("mixed_precision") in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(config.get("mixed_precision") == "fp16" and use_cuda_amp)) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=(config.get("mixed_precision") == "fp16" and use_cuda_amp))
    amp_dtype = torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16

    start_time = time.time()
    loss_history = []
    global_step = 0
    status = "OK"

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(epochs_val):
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            inputs = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                    loss = outputs.loss
            else:
                outputs = model(**inputs)
                loss = outputs.loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss encountered at step {global_step} on task {task_name}.")
                return {
                    "task": task_name,
                    "seed": seed,
                    "lr": lr_val,
                    "batch_size": bs_val,
                    "epochs": epochs_val,
                    "final_loss": float("nan"),
                    "eval_metrics": {"accuracy": 0.0, "score": 0.0},
                    "status": "FAIL_NAN",
                    "training_time": time.time() - start_time,
                    "num_params": num_trainable_params,
                    "loss_history": loss_history,
                }

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            # Majority class collapse check on binary classification
            if not is_regression and global_step >= 100 and loss_val > 0.6931:
                # If loss has not dropped below random guessing after 100 steps
                if sum(loss_history[-20:]) / 20.0 > 0.692:
                    logger.warning(f"Majority class collapse detected at step {global_step} (loss={loss_val:.4f}).")
                    status = "FAIL_COLLAPSE"

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()

            if status == "FAIL_COLLAPSE":
                break
        if status == "FAIL_COLLAPSE":
            break

    elapsed_time = time.time() - start_time

    # Run validation evaluation
    eval_metrics = evaluate_model(
        model=model,
        eval_dataset=eval_dataset,
        task_name=task_name,
        batch_size=int(config.get("eval_batch_size", 64)),
        tokenizer=tokenizer,
    )

    final_loss = loss_history[-1] if loss_history else 0.0

    return {
        "task": task_name,
        "seed": seed,
        "lr": lr_val,
        "batch_size": bs_val,
        "epochs": epochs_val,
        "final_loss": final_loss,
        "eval_metrics": eval_metrics,
        "status": status,
        "training_time": elapsed_time,
        "num_params": num_trainable_params,
        "loss_history": loss_history,
    }
