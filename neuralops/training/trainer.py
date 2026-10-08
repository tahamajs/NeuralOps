"""Training pipelines: DDP helper, MixedPrecisionTrainer with grad accumulation, LoRATrainer, and CurriculumScheduler."""
from __future__ import annotations
from typing import Callable, Any, Iterator, Iterable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = [
    "setup_ddp_training",
    "MixedPrecisionTrainer",
    "LoRATrainer",
    "CurriculumScheduler",
]


def setup_ddp_training(
    model: nn.Module,
    rank: int = 0,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wraps model in DistributedDataParallel if torch.distributed is initialized."""
    if torch.distributed.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    return model


class MixedPrecisionTrainer:
    """Full-featured trainer with automatic mixed precision, gradient accumulation, and epoch stepping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.dtype = dtype
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler("cuda", enabled=(enabled and self.device.type == "cuda"))

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Executes a single forward + backward step with AMP."""
        self.model.train()
        self.optimizer.zero_grad()

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled):
            out = self.model(x)
            loss = self.loss_fn(out, y)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return float(loss.item())

    def train_epoch(
        self,
        dataloader: Iterable[Any],
        grad_accum: int = 1,
        max_steps: int | None = None,
    ) -> float:
        """Full training loop with gradient accumulation and step budgeting."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        for step, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
                with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled):
                    out = self.model(x)
                    loss = self.loss_fn(out, y) / max(grad_accum, 1)
            elif isinstance(batch, dict):
                with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled):
                    out = self.model(**batch)
                    loss = self.loss_fn(out, batch) / max(grad_accum, 1)
            else:
                continue

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += float(loss.item()) * grad_accum
            n_batches += 1
            if max_steps is not None and (step + 1) >= max_steps:
                break

        # Flush any remaining unstepped gradients
        if n_batches % grad_accum != 0:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1)


class LoRATrainer:
    """Fine-tuning trainer that freezes base weights and optimizes only adapter parameters."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        trainable_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        if not trainable_params:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class CurriculumScheduler:
    """Progressively increases sequence length during training."""

    def __init__(
        self,
        min_seq_len: int = 128,
        max_seq_len: int = 2048,
        total_steps: int = 1000,
    ) -> None:
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.total_steps = max(total_steps, 1)

    def get_seq_len(self, current_step: int) -> int:
        progress = min(max(current_step / self.total_steps, 0.0), 1.0)
        curr = int(self.min_seq_len + progress * (self.max_seq_len - self.min_seq_len))
        return curr
