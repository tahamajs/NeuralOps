#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Data-Parallel Benchmark Executor for Accelerated Multi-GPU Evaluation.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.distributed.context import DistributedContext

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Evaluation result from DataParallelBenchmark."""

    metric_value: float
    num_samples: int
    wall_time_seconds: float
    throughput_samples_per_sec: float
    per_gpu_metrics: List[float]
    is_distributed: bool


def _to_device(obj: Any, device: Union[torch.device, str]) -> Any:
    """Recursively move tensors in container structures to target device."""
    target_dev = torch.device(device) if isinstance(device, str) else device
    if isinstance(obj, torch.Tensor):
        return obj.to(target_dev)
    if isinstance(obj, dict):
        return {k: _to_device(v, target_dev) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_device(v, target_dev) for v in obj)
    if isinstance(obj, list):
        return [_to_device(v, target_dev) for v in obj]
    return obj


def _get_num_samples(dataset: Any) -> int:
    """Determine number of samples in dataset."""
    if isinstance(dataset, dict):
        for v in dataset.values():
            if isinstance(v, (torch.Tensor, list)):
                return len(v)
    elif hasattr(dataset, "__len__"):
        return len(dataset)
    return 1


class DataParallelBenchmark:
    """
    Executes benchmark evaluations split across available GPUs in data-parallel fashion.

    Partitions dataset into world_size disjoint shards, runs metric evaluation
    on each shard on its mapped GPU, and aggregates metrics via collective reduction.
    Falls back gracefully to single-GPU or CPU execution if only 1 device is available.
    """

    def __init__(self, context: Optional[DistributedContext] = None) -> None:
        self.context = context or DistributedContext()
        if self.context.is_distributed():
            self.world_size = self.context.world_size
        elif self.context.available_gpus > 1:
            self.world_size = self.context.available_gpus
        else:
            self.world_size = 1

    @staticmethod
    def partition_dataset(dataset: Any, rank: int, world_size: int) -> Tuple[Any, int]:
        """
        Partition dataset into world_size disjoint shards.
        Rank r receives indices[r::world_size].
        """
        if world_size <= 1:
            return dataset, _get_num_samples(dataset)

        if isinstance(dataset, dict):
            shard: Dict[str, Any] = {}
            for k, v in dataset.items():
                if isinstance(v, (torch.Tensor, list)):
                    shard[k] = v[rank::world_size]
                else:
                    shard[k] = v
            return shard, _get_num_samples(shard)

        if isinstance(dataset, (torch.Tensor, list, tuple)):
            shard_seq = dataset[rank::world_size]
            return shard_seq, len(shard_seq)

        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            indices = list(range(rank, len(dataset), world_size))
            shard_items = [dataset[i] for i in indices]
            return shard_items, len(shard_items)

        return dataset, 1

    def run(
        self,
        model: nn.Module,
        dataset: Any,
        metric_fn: Callable[[nn.Module, Any], float],
        batch_size: int = 32,
        device: Optional[Union[torch.device, str]] = None,
        reduce_op: str = "mean",
    ) -> BenchmarkResult:
        """
        Run evaluation on model and dataset.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        dataset : Any
            Dataset or token tensor/dictionary to evaluate.
        metric_fn : Callable[[nn.Module, Any], float]
            Metric function accepting (model, shard) and returning scalar metric.
        batch_size : int, optional
            Evaluation batch size, by default 32.
        device : Optional[Union[torch.device, str]], optional
            Target device for single-device execution.
        reduce_op : str, optional
            Reduction operator for aggregation ("mean", "sum", "max"), by default "mean".
        """
        total_samples = _get_num_samples(dataset)

        # -------------------------------------------------------------
        # 1. Multi-process Distributed Mode (torchrun / active process group)
        # -------------------------------------------------------------
        if self.context.is_distributed():
            rank = self.context.rank
            world_size = self.context.world_size
            target_device = self.context.get_device(rank)

            model_proc = model.to(target_device)
            model_proc.eval()

            shard, shard_len = self.partition_dataset(dataset, rank, world_size)
            shard_dev = _to_device(shard, target_device)

            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            self.context.barrier()

            t0 = time.perf_counter()
            with torch.no_grad():
                local_metric = float(metric_fn(model_proc, shard_dev))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            local_time = time.perf_counter() - t0

            # Max wall time across all ranks
            time_tensor = torch.tensor([local_time], device=target_device)
            self.context.all_reduce(time_tensor, op="max")
            wall_time = float(time_tensor.item())

            # Aggregate metric
            if reduce_op.lower() in ("mean", "weighted_mean"):
                # Weighted by sample count
                weighted_val = local_metric * shard_len
                sum_tensor = torch.tensor([weighted_val, float(shard_len)], device=target_device)
                self.context.all_reduce(sum_tensor, op="sum")
                metric_value = float(sum_tensor[0].item() / max(sum_tensor[1].item(), 1.0))
            elif reduce_op.lower() == "max":
                max_tensor = torch.tensor([local_metric], device=target_device)
                self.context.all_reduce(max_tensor, op="max")
                metric_value = float(max_tensor.item())
            else:
                sum_tensor = torch.tensor([local_metric], device=target_device)
                self.context.all_reduce(sum_tensor, op="sum")
                metric_value = float(sum_tensor.item())

            # Gather per-gpu metrics
            per_gpu_t = torch.zeros(world_size, device=target_device)
            per_gpu_t[rank] = local_metric
            self.context.all_reduce(per_gpu_t, op="sum")
            per_gpu_metrics = per_gpu_t.tolist()

            throughput = total_samples / max(wall_time, 1e-9)
            return BenchmarkResult(
                metric_value=metric_value,
                num_samples=total_samples,
                wall_time_seconds=wall_time,
                throughput_samples_per_sec=throughput,
                per_gpu_metrics=per_gpu_metrics,
                is_distributed=True,
            )

        # -------------------------------------------------------------
        # 2. Multi-GPU Single Process Mode (2+ GPUs detected on host)
        # -------------------------------------------------------------
        if self.world_size > 1 and torch.cuda.is_available() and self.context.available_gpus > 1:
            num_gpus = self.context.available_gpus
            shards_and_counts = [self.partition_dataset(dataset, r, num_gpus) for r in range(num_gpus)]
            shards = [sc[0] for sc in shards_and_counts]
            counts = [sc[1] for sc in shards_and_counts]

            shards_dev = [_to_device(shards[r], f"cuda:{r}") for r in range(num_gpus)]
            model_replicas = [copy.deepcopy(model).to(f"cuda:{r}") for r in range(num_gpus)]
            for m in model_replicas:
                m.eval()

            gpu_times: List[float] = []
            per_gpu_metrics: List[float] = []

            for r in range(num_gpus):
                dev = torch.device(f"cuda:{r}")
                torch.cuda.set_device(dev)
                torch.cuda.synchronize(dev)

                t_start = time.perf_counter()
                with torch.no_grad():
                    metric_val = float(metric_fn(model_replicas[r], shards_dev[r]))
                torch.cuda.synchronize(dev)
                t_elapsed = time.perf_counter() - t_start

                gpu_times.append(t_elapsed)
                per_gpu_metrics.append(metric_val)

            # Max execution time represents the parallel throughput bottleneck
            wall_time = max(gpu_times) if gpu_times else 1e-9

            if reduce_op.lower() in ("mean", "weighted_mean"):
                metric_value = sum(per_gpu_metrics[r] * counts[r] for r in range(num_gpus)) / max(total_samples, 1)
            elif reduce_op.lower() == "max":
                metric_value = max(per_gpu_metrics)
            else:
                metric_value = sum(per_gpu_metrics)

            throughput = total_samples / max(wall_time, 1e-9)
            return BenchmarkResult(
                metric_value=metric_value,
                num_samples=total_samples,
                wall_time_seconds=wall_time,
                throughput_samples_per_sec=throughput,
                per_gpu_metrics=per_gpu_metrics,
                is_distributed=True,
            )

        # -------------------------------------------------------------
        # 3. Single Device / CPU Fallback Mode
        # -------------------------------------------------------------
        target_dev = torch.device(device) if device is not None else self.context.get_device()
        model_single = copy.deepcopy(model).to(target_dev)
        model_single.eval()
        dataset_dev = _to_device(dataset, target_dev)

        if target_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(target_dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            metric_val = float(metric_fn(model_single, dataset_dev))
        if target_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(target_dev)
        wall_time = time.perf_counter() - t0

        throughput = total_samples / max(wall_time, 1e-9)
        return BenchmarkResult(
            metric_value=metric_val,
            num_samples=total_samples,
            wall_time_seconds=wall_time,
            throughput_samples_per_sec=throughput,
            per_gpu_metrics=[metric_val],
            is_distributed=False,
        )


__all__ = ["BenchmarkResult", "DataParallelBenchmark"]
