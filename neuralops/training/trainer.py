"""Training pipelines: DDP helper, MixedPrecisionTrainer, LoRATrainer, and CurriculumScheduler."""
from __future__ import annotations
from typing import Callable, Any, Iterator
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
    """Trains PyTorch models with AMP (Automatic Mixed Precision) and GradScaler."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str | torch.device = "cpu",
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler("cuda", enabled=(enabled and self.device.type == "cuda"))

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.enabled):
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


class LoRATrainer:
    """Fine-tuning trainer that freezes base weights and optimizes only adapter parameters."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        # Freeze non-lora parameters
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
