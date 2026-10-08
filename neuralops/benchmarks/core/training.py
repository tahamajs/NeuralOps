"""Training and recovery fine-tuning loops for benchmark models."""
from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn
from torch.optim import AdamW

LOG = logging.getLogger(__name__)


def finetune_classification(
    model: nn.Module,
    train_loader: Any,
    eval_loader: Any = None,
    epochs: int = 1,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Fine-tune classification model across training epochs with early NaN stopping.

    Returns:
        Dictionary of training metadata (final_loss, epochs_completed, status).
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    step = 0
    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                LOG.warning("NaN or Inf loss encountered at step %d. Stopping training.", step)
                return {"final_loss": float("nan"), "epochs": epoch, "status": "NaN_LOSS"}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            count += 1
            step += 1

            if max_steps is not None and step >= max_steps:
                break

        avg_loss = running_loss / max(count, 1)
        epoch_losses.append(avg_loss)
        LOG.info("Epoch %d/%d completed. Loss: %.4f", epoch + 1, epochs, avg_loss)
        if max_steps is not None and step >= max_steps:
            break

    model.eval()
    return {
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epochs_completed": len(epoch_losses),
        "steps_completed": step,
        "status": "success",
    }


def recovery_finetune(
    model: nn.Module,
    train_loader: Any,
    steps: int = 50,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Execute fixed number of recovery fine-tuning steps after compression."""
    return finetune_classification(
        model=model,
        train_loader=train_loader,
        epochs=1,
        lr=lr,
        device=device,
        max_steps=steps,
    )


__all__ = [
    "finetune_classification",
    "recovery_finetune",
]
