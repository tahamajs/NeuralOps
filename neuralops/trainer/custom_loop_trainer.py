"""
High-performance custom training loop trainer with DDP gradient accumulation synchronization safety.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .metrics import get_task_metrics
from .registry import TrainerRegistry
from .utils import clip_grad_norm_factorized, evaluate_model, train_model

logger = logging.getLogger(__name__)


@TrainerRegistry.register("custom_loop")
class CustomLoopTrainer(BaseTrainer):
    """
    Custom loop trainer optimized for maximum GPU compute throughput, mixed-precision training,
    and distributed data parallel (DDP) gradient accumulation without redundant all-reduce overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: dict[str, Any],
        results_manager: Any | None = None,
        device: torch.device | None = None,
        num_labels: int = 2,
        is_multichoice: bool = False,
        compute_metrics_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.results_manager = results_manager
        self.device = device or (
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                    else "cpu"
                )
            )
        )
        self.num_labels = num_labels
        self.is_multichoice = is_multichoice
        self.compute_metrics_fn = compute_metrics_fn or get_task_metrics(
            self.config.get("task_name", "mrpc")
        )
        self.kwargs = kwargs

    def _make_loader(self, dataset: Any, shuffle: bool = False) -> DataLoader:
        """Make DataLoader optimized for high-throughput GPU training."""
        pin = self.device.type in ["cuda", "mps"]
        num_workers = int(
            self.config.get("num_workers", 4 if self.device.type == "cuda" else 0)
        )
        batch_size = int(self.config.get("batch_size", 128))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": pin,
            "num_workers": num_workers,
            "collate_fn": (
                dataset.collate_fn if hasattr(dataset, "collate_fn") else None
            ),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        return DataLoader(dataset, **loader_kwargs)

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        amp_dtype: torch.dtype = torch.float32,
        use_amp: bool = False,
        pbar: Any | None = None,
    ) -> float:
        """
        Execute a single training epoch with DDP gradient accumulation synchronization optimization.

        # FIX: Non-final micro-batches are wrapped in model.no_sync() or contextlib.nullcontext()
        # to eliminate redundant all-reduce synchronization across distributed workers during gradient accumulation.
        """
        self.model.train()
        total_epoch_loss = 0.0
        num_batches = len(train_loader)
        accum_steps = max(1, gradient_accumulation_steps)

        for step, batch in enumerate(train_loader):
            # FIX: Determine whether this micro-batch is the final step of the accumulation window or epoch
            is_sync_step = ((step + 1) % accum_steps == 0) or (step + 1 == num_batches)

            # FIX: Wrap non-final micro-batches in model.no_sync() for DDP to eliminate redundant all-reduce synchronization
            sync_context = (
                contextlib.nullcontext()
                if (
                    is_sync_step
                    or not hasattr(self.model, "no_sync")
                    or not callable(self.model.no_sync)
                )
                else self.model.no_sync()
            )

            with sync_context:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=(use_amp and self.device.type in ["cuda", "mps"]),
                ):
                    if self.is_multichoice:
                        input_ids = batch["input_ids"].to(
                            self.device, non_blocking=True
                        )
                        attention_mask = batch["attention_mask"].to(
                            self.device, non_blocking=True
                        )
                        true_labels = batch["labels"].to(self.device, non_blocking=True)
                        bs, nc, sl = input_ids.shape
                        flat_ids = input_ids.view(-1, sl)
                        flat_mask = attention_mask.view(-1, sl)
                        outputs = self.model(
                            input_ids=flat_ids, attention_mask=flat_mask
                        )
                        logits = (
                            outputs["logits"]
                            if isinstance(outputs, dict)
                            else (
                                outputs.logits
                                if hasattr(outputs, "logits")
                                else outputs
                            )
                        )
                        combined_labels = true_labels.repeat_interleave(nc)
                        loss = nn.functional.cross_entropy(logits, combined_labels)
                    else:
                        input_ids = batch["input_ids"].to(
                            self.device, non_blocking=True
                        )
                        attention_mask = batch["attention_mask"].to(
                            self.device, non_blocking=True
                        )
                        labels = batch["labels"].to(self.device, non_blocking=True)
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        logits = (
                            outputs["logits"]
                            if isinstance(outputs, dict)
                            else (
                                outputs.logits
                                if hasattr(outputs, "logits")
                                else outputs
                            )
                        )
                        loss = nn.functional.cross_entropy(logits, labels)

                    # FIX: Scale loss by accumulation factor so gradient magnitudes match the full effective batch size
                    if accum_steps > 1:
                        loss = loss / accum_steps

                if torch.isfinite(loss):
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    curr_loss = loss.detach().item() * accum_steps
                    total_epoch_loss += curr_loss
                else:
                    curr_loss = 0.0

                # FIX: Only unscale, clip gradients, and step optimizer when synchronized on accumulation boundary
                if scaler is not None and scaler.is_enabled():
                    try:
                        scaler.unscale_(optimizer)
                    except RuntimeError:
                        pass
                    clip_grad_norm_factorized(self.model, max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_factorized(self.model, max_norm=max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if pbar is not None and (
                (step + 1) % max(1, int(self.config.get("logging_steps", 10))) == 0
                or step == num_batches - 1
            ):
                pbar.set_postfix({"loss": f"{curr_loss:.4f}"})

        return total_epoch_loss / max(1, num_batches)

    def train(self) -> dict[str, Any]:
        """Fit the model on the provided dataset with full DDP and accumulation safety."""
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = self._make_loader(self.eval_dataset, shuffle=False)

        class AttrDict:
            pass

        cfg = AttrDict()
        cfg.learning_rate = float(self.config.get("learning_rate", 2e-5))
        cfg.epochs = int(self.config.get("epochs", 3))
        cfg.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        cfg.gradient_accumulation_steps = int(
            self.config.get("gradient_accumulation_steps", 1)
        )
        cfg.fp16 = bool(self.config.get("fp16", False))
        cfg.amp = bool(self.config.get("amp", True))
        cfg.logging_steps = int(self.config.get("logging_steps", 10))

        result = train_model(
            self.model,
            train_loader,
            val_loader,
            self.device,
            cfg,
            self.num_labels,
            self.is_multichoice,
        )

        if self.results_manager is not None:
            if hasattr(self.results_manager, "save_final_model"):
                self.results_manager.save_final_model(self.model)
            if hasattr(self.results_manager, "save_plots"):
                self.results_manager.save_plots()

        return result

    def evaluate(self) -> dict[str, Any]:
        """Run evaluate and return results."""
        val_loader = self._make_loader(self.eval_dataset, shuffle=False)
        return evaluate_model(
            self.model,
            val_loader,
            self.device,
            self.num_labels,
            self.is_multichoice,
            compute_all_metrics=True,
        )
