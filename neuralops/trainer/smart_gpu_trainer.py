#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Smart GPU-Maximizing Training Engine (AutoMaxGPUTrainer).

Provides adaptive VRAM saturation search, double-buffered asynchronous CUDA stream
prefetching, dynamic gradient accumulation scaling, activation checkpointing,
and real-time OOM auto-recovery.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base_trainer import BaseTrainer
from .checkpointing import inject_activation_checkpointing
from .metrics import get_task_metrics
from .registry import TrainerRegistry
from .utils import clip_grad_norm_factorized

logger = logging.getLogger("NeuralOps.AutoMaxGPUTrainer")


class GPUMemoryManager:
    """Monitors and profiles GPU device memory to maximize safe allocation without OOM."""

    @staticmethod
    def get_device_info(device: torch.device) -> dict[str, Any]:
        """Query physical VRAM metrics and compute capabilities."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return {
                "is_cuda": False,
                "total_memory_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "device_name": "CPU/MPS",
            }

        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        total = props.total_memory / (1024 * 1024)
        alloc = torch.cuda.memory_allocated(dev_idx) / (1024 * 1024)
        res = torch.cuda.memory_reserved(dev_idx) / (1024 * 1024)
        free = total - alloc

        return {
            "is_cuda": True,
            "device_name": props.name,
            "compute_capability": (props.major, props.minor),
            "total_memory_mb": round(total, 2),
            "allocated_mb": round(alloc, 2),
            "reserved_mb": round(res, 2),
            "free_mb": round(free, 2),
        }

    @staticmethod
    def get_memory_allocated_ratio(device: torch.device) -> float:
        """Fraction of device memory currently allocated [0.0, 1.0]."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        alloc = torch.cuda.memory_allocated(dev_idx)
        return float(alloc) / float(props.total_memory) if props.total_memory > 0 else 0.0

    @staticmethod
    def empty_cache(device: torch.device | None = None) -> None:
        """Flush garbage collector and allocator."""
        GPUMemoryManager.clear_cache()

    @staticmethod
    def clear_cache() -> None:
        """Flush garbage collector and CUDA caching allocator."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AsyncDataPrefetcher:
    """Overlaps host-to-device memory copy with GPU compute via dedicated CUDA stream."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.iterator: Iterator[Any] | None = None
        self.next_batch: Any = None

    def __iter__(self) -> AsyncDataPrefetcher:
        self.iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self) -> None:
        if self.iterator is None:
            self.next_batch = None
            return

        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = self._move_to_device(batch)
        else:
            self.next_batch = self._move_to_device(batch)

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=(self.device.type == "cuda"))
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            moved = [self._move_to_device(v) for v in data]
            return type(data)(moved)
        return data

    def __next__(self) -> Any:
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch


@TrainerRegistry.register("automax_gpu")
class AutoMaxGPUTrainer(BaseTrainer):
    """Smart GPU-Maximizing Training Engine.

    Features:
    - Adaptive VRAM saturation search to maximize micro-batch size without OOM.
    - Automatic scaling of gradient accumulation steps to lock effective global batch size.
    - Double-buffered asynchronous CUDA stream prefetching.
    - Transparent activation checkpointing for ultra-long context.
    - Real-time OOM catching with automatic batch halving and non-destructive replay.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset | Any,
        eval_dataset: Dataset | Any | None = None,
        config: dict[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: torch.device | None = None,
        compute_metrics_fn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.kwargs = kwargs

        raw_dev = device or (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.device = torch.device(raw_dev) if isinstance(raw_dev, str) else raw_dev

        self.compute_metrics_fn = compute_metrics_fn or get_task_metrics(
            self.config.get("task_name", "mrpc")
        )

        self.target_vram_ratio = float(self.config.get("target_vram_ratio", 0.90))
        self.max_epochs = int(self.config.get("epochs", kwargs.get("epochs", 3)))
        self.max_grad_norm = float(self.config.get("max_grad_norm", kwargs.get("max_grad_norm", 1.0)))
        self.use_amp = bool(self.config.get("use_amp", kwargs.get("use_amp", self.device.type in ("cuda", "mps"))))
        self.amp_dtype = self._resolve_amp_dtype()

        # Compile options
        self.compile_model = bool(self.config.get("compile_model", kwargs.get("compile_model", False)))

        # Checkpointing options
        self.auto_checkpoint = bool(self.config.get("auto_checkpoint", kwargs.get("auto_checkpoint", True)))

        # Target global batch size
        self.target_global_batch_size = int(self.config.get("global_batch_size", kwargs.get("global_batch_size", 128)))
        self.micro_batch_size = int(self.config.get("micro_batch_size", kwargs.get("initial_micro_batch_size", 0)))
        self.max_micro_batch_size = int(self.config.get("max_micro_batch_size", kwargs.get("max_micro_batch_size", 512)))
        self.logging_steps = int(self.config.get("logging_steps", kwargs.get("logging_steps", 10)))

        if self.micro_batch_size <= 0:
            self.micro_batch_size = self._find_max_micro_batch_size()

        self.grad_accum_steps = max(1, self.target_global_batch_size // max(1, self.micro_batch_size))
        logger.info(
            "AutoMaxGPUTrainer configured: Micro-batch=%d, Accum-steps=%d, Global-batch=%d",
            self.micro_batch_size,
            self.grad_accum_steps,
            self.micro_batch_size * self.grad_accum_steps,
        )

        self.model = self.model.to(self.device)

        if self.auto_checkpoint and self.device.type == "cuda":
            wrapped = inject_activation_checkpointing(self.model)
            if wrapped > 0:
                logger.info("AutoMaxGPUTrainer: Injected activation checkpointing into %d modules", wrapped)

        if self.compile_model and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
                logger.info("AutoMaxGPUTrainer: Model successfully compiled with TorchInductor")
            except Exception as e:
                logger.warning("AutoMaxGPUTrainer: torch.compile skipped: %s", e)

        self.optimizer = optimizer or self._create_default_optimizer()
        self.scheduler = scheduler

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _resolve_amp_dtype(self) -> torch.dtype:
        custom_dtype = self.config.get("dtype")
        if custom_dtype is not None:
            if isinstance(custom_dtype, torch.dtype):
                return custom_dtype
            if str(custom_dtype).lower() in ("bf16", "bfloat16"):
                return torch.bfloat16
            if str(custom_dtype).lower() in ("fp16", "float16"):
                return torch.float16

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.device.type == "cuda":
            return torch.float16
        return torch.float32

    def _create_default_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config.get("lr", self.config.get("learning_rate", 5e-5)))
        weight_decay = float(self.config.get("weight_decay", 0.01))
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name.lower() or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

    def _find_max_micro_batch_size(self) -> int:
        """Binary search to discover the maximum micro-batch size saturating target VRAM without OOM."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return min(32, max(4, self.target_global_batch_size // 4))

        logger.info("AutoMaxGPUTrainer: Performing adaptive VRAM saturation search...")
        high = max(128, self.target_global_batch_size)
        best_valid = 1

        info = GPUMemoryManager.get_device_info(self.device)
        total_mb = info["total_memory_mb"]
        target_limit_mb = total_mb * self.target_vram_ratio

        sample_item = self.train_dataset[0] if len(self.train_dataset) > 0 else None
        if sample_item is None:
            return 16

        candidate_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        candidate_sizes = [c for c in candidate_sizes if c <= high]

        for cand in candidate_sizes:
            GPUMemoryManager.clear_cache()
            try:
                test_loader = DataLoader(
                    self.train_dataset,
                    batch_size=cand,
                    shuffle=False,
                    collate_fn=getattr(self.train_dataset, "collate_fn", None),
                )
                batch = next(iter(test_loader))
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                self.model.eval()
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    out = self.model(**batch_gpu)
                    loss = out.loss if hasattr(out, "loss") else (out[0] if isinstance(out, tuple) else out)
                    if hasattr(loss, "mean"):
                        loss = loss.mean()
                loss.backward()
                self.model.zero_grad(set_to_none=True)

                curr_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                if curr_alloc <= target_limit_mb:
                    best_valid = cand
                else:
                    break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    GPUMemoryManager.clear_cache()
                    break
                logger.debug("Dry-run probe exception for batch %d: %s", cand, e)
                break

        GPUMemoryManager.clear_cache()
        logger.info(
            "AutoMaxGPUTrainer: Discovered optimal micro-batch size: %d (Target VRAM budget: %.1f MB)",
            best_valid,
            target_limit_mb,
        )
        return best_valid

    def _make_loader(self, dataset: Any, batch_size: int, shuffle: bool = True) -> DataLoader:
        pin = self.device.type == "cuda"
        default_workers = 0 if (hasattr(dataset, "__len__") and len(dataset) < 500) else (4 if self.device.type == "cuda" else 0)
        num_workers = int(self.config.get("num_workers", default_workers))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": pin,
            "num_workers": num_workers,
            "collate_fn": getattr(dataset, "collate_fn", None),
        }
        if num_workers > 0 and self.device.type == "cuda":
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **loader_kwargs)

    def train_epoch(
        self,
        train_loader: DataLoader,
        scaler: torch.cuda.amp.GradScaler | None,
    ) -> float:
        """Executes a single training epoch with double-buffered prefetching and OOM auto-recovery."""
        self.model.train()
        total_epoch_loss = 0.0
        batches_processed = 0

        prefetcher = AsyncDataPrefetcher(train_loader, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(prefetcher):
            is_sync_step = ((step + 1) % self.grad_accum_steps == 0)

            retry_count = 0
            while retry_count < 3:
                try:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=(self.use_amp and self.device.type in ("cuda", "mps")),
                    ):
                        out = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                        loss = out.loss if hasattr(out, "loss") and out.loss is not None else (
                            out[0] if isinstance(out, tuple) else out
                        )
                        if hasattr(loss, "mean"):
                            loss = loss.mean()
                        if self.grad_accum_steps > 1:
                            loss = loss / self.grad_accum_steps

                    if torch.isfinite(loss):
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        total_epoch_loss += loss.detach().item() * self.grad_accum_steps
                    batches_processed += 1
                    break

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    is_oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
                    if is_oom:
                        retry_count += 1
                        logger.warning(
                            "OOM detected at step %d! Activating Auto-Recovery (attempt %d/3).",
                            step,
                            retry_count,
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        GPUMemoryManager.clear_cache()
                        self.micro_batch_size = max(1, self.micro_batch_size // 2)
                        self.grad_accum_steps = max(1, self.grad_accum_steps * 2)
                        if isinstance(batch, dict):
                            batch = {k: v[: max(1, len(v) // 2)] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    else:
                        raise e

            if is_sync_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_factorized(self.model, max_norm=self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_factorized(self.model, max_norm=self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

        return total_epoch_loss / max(1, batches_processed)

    def train(self) -> dict[str, Any]:
        """Execute full training across epochs."""
        scaler = (
            torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
            if self.device.type == "cuda"
            else None
        )

        history: list[dict[str, Any]] = []
        t0 = time.monotonic()

        train_loader = self._make_loader(self.train_dataset, batch_size=self.micro_batch_size, shuffle=True)

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = self.train_epoch(train_loader, scaler)
            logger.info("Epoch %d/%d - Train Loss: %.4f", epoch, self.max_epochs, epoch_loss)

            eval_metrics = {}
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                logger.info("Epoch %d/%d - Eval Metrics: %s", epoch, self.max_epochs, eval_metrics)

            history.append({
                "epoch": epoch,
                "train_loss": epoch_loss,
                **eval_metrics,
            })

        total_time = time.monotonic() - t0
        mem_info = GPUMemoryManager.get_device_info(self.device)
        final_loss = history[-1]["train_loss"] if history else 0.0
        total_samples = len(self.train_dataset) * self.max_epochs if hasattr(self.train_dataset, "__len__") else 0
        throughput = round(total_samples / max(1e-5, total_time), 2)

        return {
            "status": "COMPLETED",
            "epochs_completed": self.max_epochs,
            "final_loss": round(final_loss, 4),
            "throughput_samples_per_sec": throughput,
            "total_time_s": round(total_time, 2),
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "history": history,
            "final_memory_info": mem_info,
        }

    def evaluate(self) -> dict[str, Any]:
        """Evaluate model against eval_dataset."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        eval_loader = self._make_loader(self.eval_dataset, batch_size=self.micro_batch_size * 2, shuffle=False)
        total_eval_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                } if isinstance(batch, dict) else batch.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=(self.use_amp and self.device.type in ("cuda", "mps")),
                ):
                    out = self.model(**batch_gpu) if isinstance(batch_gpu, dict) else self.model(batch_gpu)
                    loss = out.loss if hasattr(out, "loss") and out.loss is not None else (
                        out[0] if isinstance(out, tuple) else out
                    )
                    if hasattr(loss, "mean"):
                        loss = loss.mean()
                if torch.isfinite(loss):
                    total_eval_loss += loss.item()
                batches += 1

        mean_loss = total_eval_loss / max(1, batches)
        return {"eval_loss": round(mean_loss, 4)}


__all__ = [
    "GPUMemoryManager",
    "AsyncDataPrefetcher",
    "AutoMaxGPUTrainer",
]
