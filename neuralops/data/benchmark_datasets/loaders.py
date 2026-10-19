#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Offline and torch-compatible benchmark dataset loaders."""

from torch.utils.data import DataLoader, Dataset

from .manifest import BenchmarkDatasetRegistry


class BenchmarkTorchDataset(Dataset):
    """Minimal Dataset wrapper preserving dictionary samples."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def load_benchmark_dataset(
    dataset_name: str,
    *,
    use_synthetic: bool = False,
    max_samples: int | None = None,
    return_dataloader: bool = False,
    seed: int = 0,
    **kwargs,
):
    """Load a benchmark, using deterministic samples when requested."""
    del kwargs
    info = BenchmarkDatasetRegistry.get(dataset_name)
    if info is None:
        if use_synthetic:
            samples = [
                {"text": f"Synthetic sample {i} from {dataset_name}", "label": i % 2}
                for i in range(max_samples or 100)
            ]
            if return_dataloader:
                return DataLoader(BenchmarkTorchDataset(samples), batch_size=1)
            return samples
        raise KeyError(f"Unknown benchmark dataset: {dataset_name}")
    count = max_samples or 100
    if use_synthetic or info.synthetic_generator is not None:
        samples = info.synthetic_generator(count, seed) if info.synthetic_generator else []
    else:
        raise RuntimeError("remote dataset loading is disabled; pass use_synthetic=True")
    if return_dataloader:
        return DataLoader(BenchmarkTorchDataset(samples), batch_size=1)
    return samples
