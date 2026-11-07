#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Distributed Context Foundation for Multi-GPU Execution.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

from neuralops.distributed.errors import E_DIST_COMM_FAILED, DistributedError

logger = logging.getLogger(__name__)


class DistributedContext:
    """Manages distributed environment discovery, process group lifecycle, and communication primitives."""

    def __init__(
        self,
        backend: str = "nccl",
        init_if_available: bool = True,
    ) -> None:
        self.available_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.world_size: int = 1
        self.rank: int = 0
        self.local_rank: int = 0
        self.backend: str = backend
        self._initialized_by_us: bool = False

        if not dist.is_available():
            logger.warning("torch.distributed is not available. Falling back to single-device mode.")
            return

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            self.backend = dist.get_backend()
            return

        if init_if_available:
            chosen_backend = backend
            if chosen_backend == "nccl" and (not torch.cuda.is_available() or not dist.is_nccl_available()):
                chosen_backend = "gloo"

            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                try:
                    dist.init_process_group(backend=chosen_backend)
                    self._initialized_by_us = True
                    self.world_size = dist.get_world_size()
                    self.rank = dist.get_rank()
                    self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
                    self.backend = chosen_backend
                except Exception as e:
                    logger.warning(
                        "Failed to initialize distributed process group: %s. Falling back to single-device mode.", e
                    )
            elif self.available_gpus > 1:
                logger.warning(
                    "Multiple GPUs detected (%d), but distributed launcher environment variables (RANK, WORLD_SIZE) "
                    "are not set. Falling back to single-device mode.",
                    self.available_gpus,
                )

    def get_device(self, rank: int | None = None) -> torch.device:
        """Return the target torch.device for the local process or specified rank."""
        if not torch.cuda.is_available() or self.available_gpus == 0:
            return torch.device("cpu")
        target_rank = self.local_rank if rank is None else rank
        if self.available_gpus > 0:
            target_rank = target_rank % self.available_gpus
        return torch.device(f"cuda:{target_rank}")

    def is_main_process(self) -> bool:
        """Return True if this is the rank-0 coordinator process."""
        return self.rank == 0

    def is_distributed(self) -> bool:
        """Return True if running in an active multi-process distributed group."""
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def barrier(self) -> None:
        """Synchronize all processes in the distributed group, or no-op if single process."""
        if self.is_distributed():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce a tensor across all distributed workers, or no-op if single process."""
        if not self.is_distributed():
            return tensor

        op_lower = op.lower()
        if op_lower in ("sum", "add"):
            reduce_op = dist.ReduceOp.SUM
        elif op_lower in ("mean", "avg"):
            reduce_op = dist.ReduceOp.AVG
        elif op_lower == "max":
            reduce_op = dist.ReduceOp.MAX
        elif op_lower == "min":
            reduce_op = dist.ReduceOp.MIN
        elif op_lower in ("product", "prod"):
            reduce_op = dist.ReduceOp.PRODUCT
        else:
            raise DistributedError(f"Unsupported reduction operator: {op}", code=E_DIST_COMM_FAILED)

        try:
            dist.all_reduce(tensor, op=reduce_op)
            return tensor
        except Exception as e:
            raise DistributedError(f"all_reduce failed: {e}", code=E_DIST_COMM_FAILED) from e

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from rank `src` to all workers, or no-op if single process."""
        if not self.is_distributed():
            return tensor

        try:
            dist.broadcast(tensor, src=src)
            return tensor
        except Exception as e:
            raise DistributedError(f"broadcast from rank {src} failed: {e}", code=E_DIST_COMM_FAILED) from e

    def cleanup(self) -> None:
        """Destroy the distributed process group if initialized by this context."""
        if self._initialized_by_us and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized_by_us = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0


__all__ = ["DistributedContext"]
