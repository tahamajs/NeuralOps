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
    mixed_precision: str = "no",
    distributed_backend: str = "none",
    tracker: Any = None,
    log_interval: int = 10,
) -> dict[str, Any]:
    """
    Fine-tune classification model across training epochs with mixed-precision,
    early NaN stopping, and real-time MLOps step-level telemetry.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Distributed backend wrapping via accelerate
    dist_backend = (distributed_backend or "none").lower()
    if dist_backend in ("ddp", "fsdp", "deepspeed"):
        try:
            from accelerate import Accelerator
            accelerator = Accelerator(mixed_precision=mixed_precision if mixed_precision in ("fp16", "bf16") else "no")
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
            device = accelerator.device
        except Exception as e:
            LOG.warning("Distributed backend '%s' requested but Accelerate initialization failed: %s. Using local device.", dist_backend, e)

    # Mixed precision setup
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    mp_type = (mixed_precision or "no").lower()
    amp_enabled = mp_type in ("fp16", "bf16") and use_cuda
    amp_dtype = torch.bfloat16 if mp_type == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_type == "fp16" and use_cuda))


    step = 0
    epoch_losses = []
    import sys
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            file=sys.stderr,
            leave=False,
        ) if has_tqdm else train_loader

        for batch in pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                LOG.warning("NaN or Inf loss encountered at step %d. Stopping training.", step)
                return {"final_loss": float("nan"), "epochs": epoch, "status": "NaN_LOSS"}

            if amp_enabled and mp_type == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            cur_loss = float(loss.item())
            running_loss += cur_loss
            count += 1
            step += 1

            if tracker is not None and step % log_interval == 0:
                try:
                    tracker.log_metrics({"train/loss": cur_loss, "train/epoch": epoch + (count / max(len(train_loader), 1))}, step=step)
                except Exception:
                    pass

            LOG_EVERY = 50
            if has_tqdm and hasattr(pbar, "set_postfix"):
                if step % LOG_EVERY == 0 or (len(train_loader) and count == len(train_loader)):
                    pbar.set_postfix({"loss": f"{cur_loss:.4f}", "avg_loss": f"{running_loss / count:.4f}"})

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


def train_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Any,
    epochs: int = 1,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
    temperature: float = 2.0,
    alpha: float = 0.5,
    max_steps: int | None = None,
    mixed_precision: str = "no",
    tracker: Any = None,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Execute Knowledge Distillation training loop from teacher to student with mixed precision & MLOps."""
    from neuralops.benchmarks.core.kd import KDLoss, kd_train_step

    kd_loss_fn = KDLoss(temperature=temperature, alpha=alpha)
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    mp_type = (mixed_precision or "no").lower()
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_type == "fp16" and use_cuda))

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for batch in train_loader:
            step_loss = kd_train_step(
                student=student,
                teacher=teacher,
                batch=batch,
                optimizer=optimizer,
                kd_loss_fn=kd_loss_fn,
                device=device,
                mixed_precision=mixed_precision,
                scaler=scaler,
            )
            running_loss += step_loss
            count += 1
            step += 1

            if tracker is not None and step % log_interval == 0:
                try:
                    tracker.log_metrics({"kd/loss": step_loss, "kd/epoch": epoch + (count / max(len(train_loader), 1))}, step=step)
                except Exception:
                    pass

            if max_steps is not None and step >= max_steps:
                break
        avg_loss = running_loss / max(count, 1)
        epoch_losses.append(avg_loss)
        LOG.info("KD Epoch %d/%d completed. Loss: %.4f", epoch + 1, epochs, avg_loss)
        if max_steps is not None and step >= max_steps:
            break

    student.eval()
    return {
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epochs_completed": len(epoch_losses),
        "steps_completed": step,
        "training_type": "kd",
        "status": "success",
    }


def recovery_finetune(
    model: nn.Module,
    train_loader: Any,
    steps: int = 50,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
    mixed_precision: str = "no",
    tracker: Any = None,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Execute fixed number of recovery fine-tuning steps after compression."""
    return finetune_classification(
        model=model,
        train_loader=train_loader,
        epochs=1,
        lr=lr,
        device=device,
        max_steps=steps,
        mixed_precision=mixed_precision,
        tracker=tracker,
        log_interval=log_interval,
    )


def train_with_training_type(
    model: nn.Module,
    train_loader: Any,
    training_type: str = "finetune",
    epochs: int = 1,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
    max_steps: int | None = None,
    teacher: nn.Module | None = None,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.5,
    mixed_precision: str = "no",
    tracker: Any = None,
    log_interval: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute training according to configured training_type ('none', 'finetune', 'kd', 'recovery', 'lora')."""
    t_type = (training_type or "finetune").lower()

    if t_type in ("none", "eval_only", "zero_shot"):
        return {"status": "skipped", "training_type": "none", "final_loss": 0.0}

    elif t_type in ("kd", "distillation", "knowledge_distillation"):
        if teacher is None:
            LOG.warning("KD requested but no teacher provided. Falling back to standard fine-tuning.")
            return finetune_classification(
                model=model,
                train_loader=train_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                max_steps=max_steps,
                mixed_precision=mixed_precision,
                distributed_backend=kwargs.get("distributed_backend", "none"),
                tracker=tracker,
                log_interval=log_interval,
            )
        return train_kd(
            student=model,
            teacher=teacher,
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            temperature=kd_temperature,
            alpha=kd_alpha,
            max_steps=max_steps,
            mixed_precision=mixed_precision,
            tracker=tracker,
            log_interval=log_interval,
        )

    elif t_type in ("recovery", "fast_recovery"):
        steps = max_steps or kwargs.get("recovery_steps", 50)
        return recovery_finetune(
            model=model,
            train_loader=train_loader,
            steps=steps,
            lr=lr,
            device=device,
            mixed_precision=mixed_precision,
            tracker=tracker,
            log_interval=log_interval,
        )

    else:
        # Standard full or fine-tuning
        return finetune_classification(
            model=model,
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            max_steps=max_steps,
            mixed_precision=mixed_precision,
            distributed_backend=kwargs.get("distributed_backend", "none"),
            tracker=tracker,
            log_interval=log_interval,
        )



__all__ = [
    "finetune_classification",
    "recovery_finetune",
    "train_kd",
    "train_with_training_type",
]
