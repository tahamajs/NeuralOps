"""
Load‑Balanced Data Loader – inspired by Bagua’s `load_balancing_data_loader`.

When samples have different computational complexities (e.g. variable‑length
sequences in NLP), this loader distributes batches so that each GPU does
roughly the same amount of work.

References:
  - Bagua: https://github.com/BaguaSys/bagua
  - FastSpeed heterogeneous GPU data partitioning
"""

from typing import Callable, Iterator, List, Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler


class WorkloadSampler(Sampler):
    """
    Custom sampler that groups samples by computational cost so that
    each batch has a balanced total workload.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cost_fn: Optional[Callable[[int], float]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            dataset: PyTorch Dataset
            batch_size: target batch size
            cost_fn: function that returns the computational cost of sample i
                     (default: cost = 1 for all samples → standard batching)
            shuffle: whether to shuffle
            drop_last: drop incomplete last batch
            seed: random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.cost_fn = cost_fn or (lambda i: 1)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Pre‑compute cost for every sample
        self.costs = np.array(
            [self.cost_fn(i) for i in range(len(dataset))], dtype=np.float64
        )
        self._build_batches()

    def _build_batches(self):
        """Create balanced batches using a greedy bin‑packing algorithm."""
        indices = np.arange(len(self.costs))
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(indices)

        # Sort by cost (descending) for better packing
        sorted_order = np.argsort(-self.costs[indices])
        indices = indices[sorted_order]

        batches = []
        current_batch = []
        current_cost = 0.0

        for idx in indices:
            c = self.costs[idx]
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_cost = 0.0
            current_batch.append(idx)
            current_cost += c

        if current_batch and (
            not self.drop_last or len(current_batch) == self.batch_size
        ):
            batches.append(current_batch)

        self.batches = batches

    def __iter__(self) -> Iterator[List[int]]:
        self.epoch += 1
        self._build_batches()
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class LoadBalancedDataLoader(DataLoader):
    """
    Drop‑in replacement for PyTorch `DataLoader` that uses workload‑aware
    batching to keep all GPUs busy for roughly the same amount of time.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        cost_fn: Optional[Callable[[int], float]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        sampler = WorkloadSampler(
            dataset=dataset,
            batch_size=batch_size,
            cost_fn=cost_fn,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        super().__init__(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )


class BatchSizeBalancer:
    """
    Dynamically adjusts per‑GPU batch size based on GPU capability.

    Inspired by FastSpeed: more powerful GPUs get larger batches.

    Usage:
        balancer = BatchSizeBalancer(gpu_infos)
        per_gpu_batch = balancer.compute_batch_sizes(global_batch_size=128)
    """

    def __init__(self, gpu_infos: List):
        """
        Args:
            gpu_infos: list of GPUInfo objects from the GPUAllocator
        """
        # Compute relative power based on free memory
        memories = [g.free_memory_mb for g in gpu_infos]
        total = sum(memories)
        if total == 0:
            self.weights = [1.0 / len(gpu_infos)] * len(gpu_infos)
        else:
            self.weights = [m / total for m in memories]

    def compute_batch_sizes(self, global_batch_size: int) -> List[int]:
        """
        Split `global_batch_size` across GPUs according to their capability.

        Returns list of per‑GPU batch sizes that sum to global_batch_size.
        """
        raw = [max(1, int(global_batch_size * w)) for w in self.weights]
        # Adjust to sum exactly to global_batch_size
        diff = global_batch_size - sum(raw)
        # Add remainder to the most capable GPU
        best_idx = max(range(len(raw)), key=lambda i: self.weights[i])
        raw[best_idx] += diff
        return raw
