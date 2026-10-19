#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""


NeuralOps Multi-Dataset Streaming Loader.

Provides unified streaming and map-style dataset loading across multiple
text and token corpora with dynamic round-robin and weighted interleaving,
per-dataset subset parsing, and robust offline fallback generators.
"""

from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger("NeuralOps.MultiDataset")


class _FallbackSyntheticDataset:
    """Robust offline fallback synthetic dataset generator.

    Generates synthetic text / token sequences when external dataset repositories
    or network caches are offline or unreachable.
    """

    def __init__(
        self,
        name: str,
        subset: Optional[str] = None,
        num_samples: int = 1000,
        streaming: bool = True,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.subset = subset or "default"
        self.num_samples = num_samples
        self.streaming = streaming
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed + hash(self.name))
        idx = 0
        while True:
            if not self.streaming and idx >= self.num_samples:
                break
            # Deterministic synthetic sample
            vocab_sample = [rng.randint(32, 126) for _ in range(32)]
            chars = "".join(chr(c) for c in vocab_sample)
            doc = {
                "text": f"NeuralOps synthetic corpus sample {idx} from {self.name} [{self.subset}]: {chars}",
                "dataset_name": self.name,
                "subset": self.subset,
                "sample_idx": idx,
            }
            yield doc
            idx += 1

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming fallback dataset has no defined length.")
        return self.num_samples


class MultiDatasetLoader(IterableDataset):
    """Multi-Dataset Streaming and Interleaving Loader for NeuralOps Pretraining.

    Supports comma-separated dataset specs, per-dataset subsets, streaming
    iterables, weighted and round-robin interleaving, and offline fallback.
    """

    def __init__(
        self,
        dataset_names: Union[str, Sequence[Union[str, Any]]],
        subsets: Optional[Union[str, Sequence[Optional[str]], Dict[str, Optional[str]]]] = None,
        weights: Optional[Union[str, Sequence[float]]] = None,
        streaming: bool = True,
        split: str = "train",
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        columns: Optional[Sequence[str]] = None,
        fallback_samples_per_dataset: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        parsed_names, parsed_subsets, parsed_weights = self.parse_dataset_spec(
            dataset_names=dataset_names,
            subsets=subsets,
            weights=weights,
        )
        self.dataset_names = parsed_names
        self.subsets = parsed_subsets
        self.weights = parsed_weights
        self.streaming = streaming
        self.split = split
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.seed = seed
        self.stopping_strategy = stopping_strategy
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = list(columns) if columns is not None else None
        self.fallback_samples_per_dataset = fallback_samples_per_dataset
        self.kwargs = kwargs

        # Materialize or configure underlying dataset sources
        self.sources: List[Any] = []
        for name in self.dataset_names:
            if not isinstance(name, str):
                # Direct dataset or iterable instance passed for testing/custom workflows
                self.sources.append(name)
            else:
                subset = self.subsets.get(name, None)
                ds = self._create_dataset_source(
                    name=name,
                    subset=subset,
                    streaming=self.streaming,
                    split=self.split,
                    cache_dir=self.cache_dir,
                    fallback_samples=self.fallback_samples_per_dataset,
                    seed=self.seed,
                    **self.kwargs,
                )
                self.sources.append(ds)

    @classmethod
    def parse_dataset_spec(
        cls,
        dataset_names: Union[str, Sequence[Union[str, Any]]],
        subsets: Optional[Union[str, Sequence[Optional[str]], Dict[str, Optional[str]]]] = None,
        weights: Optional[Union[str, Sequence[float]]] = None,
    ) -> tuple[List[Any], Dict[str, Optional[str]], List[float]]:
        """Parse raw dataset names, per-dataset subsets, and sampling weights.

        Args:
            dataset_names: Comma-separated dataset string or sequence of dataset names/objects.
            subsets: Comma-separated string, list, dictionary, or None.
            weights: Comma-separated floats, sequence of floats, or None.

        Returns:
            Tuple of (parsed_names, parsed_subsets_dict, normalized_weights_list).
        """
        # 1. Parse dataset names
        if isinstance(dataset_names, str):
            parsed_names: List[Any] = [s.strip() for s in dataset_names.split(",") if s.strip()]
        else:
            parsed_names = list(dataset_names)

        if not parsed_names:
            raise ValueError("dataset_names must contain at least one valid dataset specification.")

        num_ds = len(parsed_names)

        # 2. Parse subsets
        parsed_subsets: Dict[str, Optional[str]] = {}
        if isinstance(subsets, dict):
            for name in parsed_names:
                key = str(name)
                parsed_subsets[key] = subsets.get(key, None)
        elif isinstance(subsets, str):
            sub_list = [s.strip() if s.strip() else None for s in subsets.split(",")]
            for idx, name in enumerate(parsed_names):
                key = str(name)
                parsed_subsets[key] = sub_list[idx] if idx < len(sub_list) else None
        elif isinstance(subsets, (list, tuple)):
            for idx, name in enumerate(parsed_names):
                key = str(name)
                parsed_subsets[key] = subsets[idx] if idx < len(subsets) else None
        else:
            for name in parsed_names:
                parsed_subsets[str(name)] = None

        # 3. Parse weights
        if weights is None:
            raw_weights = [1.0 / num_ds] * num_ds
        elif isinstance(weights, str):
            raw_weights = [float(w.strip()) for w in weights.split(",") if w.strip()]
            if len(raw_weights) != num_ds:
                raise ValueError(
                    f"Number of weights ({len(raw_weights)}) must match number of datasets ({num_ds})."
                )
        else:
            raw_weights = [float(w) for w in weights]
            if len(raw_weights) != num_ds:
                raise ValueError(
                    f"Number of weights ({len(raw_weights)}) must match number of datasets ({num_ds})."
                )

        weight_sum = sum(raw_weights)
        if weight_sum <= 0.0:
            raise ValueError("Dataset sampling weights must sum to a strictly positive value.")
        normalized_weights = [w / weight_sum for w in raw_weights]

        return parsed_names, parsed_subsets, normalized_weights

    @staticmethod
    def _create_dataset_source(
        name: str,
        subset: Optional[str],
        streaming: bool,
        split: str,
        cache_dir: Optional[str],
        fallback_samples: int,
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """Attempt loading via external datasets library with robust fallback."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            ds_kwargs: Dict[str, Any] = {
                "path": name,
                "split": split,
                "streaming": streaming,
            }
            if subset is not None:
                ds_kwargs["name"] = subset
            if cache_dir is not None:
                ds_kwargs["cache_dir"] = cache_dir

            # Pass additional compatible kwargs
            for k, v in kwargs.items():
                if k not in ds_kwargs:
                    ds_kwargs[k] = v

            return load_dataset(**ds_kwargs)
        except Exception as exc:
            logger.info(
                "Initializing offline synthetic fallback for dataset '%s' (subset=%s) [reason: %s]",
                name,
                subset,
                exc,
            )
            return _FallbackSyntheticDataset(
                name=name,
                subset=subset,
                num_samples=fallback_samples,
                streaming=streaming,
                seed=seed,
            )

    @classmethod
    def interleave_datasets(
        cls,
        iterables: Sequence[Iterable[Any]],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",
    ) -> Iterator[Any]:
        """Interleave multiple dataset iterables using weighted or round-robin sampling.

        Args:
            iterables: Collection of iterable dataset streams.
            weights: Optional sampling probabilities corresponding to each stream.
            seed: Pseudo-random seed for deterministic weighted interleaving.
            stopping_strategy: 'all_exhausted' (stops when all streams finish) or
                               'first_exhausted' (stops when any stream finishes).

        Yields:
            Elements interleaved across the provided streams.
        """
        if not iterables:
            return

        active_iterators = [iter(it) for it in iterables]
        num_streams = len(active_iterators)

        # Check if pure round-robin (no weights or equal weights)
        is_uniform = weights is None or (
            len(weights) == num_streams and max(weights) - min(weights) < 1e-7
        )

        if is_uniform:
            # Round-robin interleaving
            stream_indices = list(range(num_streams))
            while stream_indices:
                exhausted_indices: List[int] = []
                for idx in stream_indices:
                    try:
                        item = next(active_iterators[idx])
                        yield item
                    except StopIteration:
                        if stopping_strategy == "first_exhausted":
                            return
                        exhausted_indices.append(idx)

                for ex_idx in exhausted_indices:
                    stream_indices.remove(ex_idx)
            return

        # Weighted sampling interleaving
        rng = random.Random(seed)
        assert weights is not None
        current_weights = list(weights)
        active_indices = list(range(num_streams))

        while active_indices:
            # Normalize active weights
            active_weights = [current_weights[i] for i in active_indices]
            total_w = sum(active_weights)
            if total_w <= 0.0:
                break
            norm_probs = [w / total_w for w in active_weights]

            # Choose one stream index according to normalized weights
            chosen_active_idx = rng.choices(range(len(active_indices)), weights=norm_probs, k=1)[0]
            chosen_stream_idx = active_indices[chosen_active_idx]

            try:
                item = next(active_iterators[chosen_stream_idx])
                yield item
            except StopIteration:
                if stopping_strategy == "first_exhausted":
                    return
                # Remove exhausted stream
                active_indices.pop(chosen_active_idx)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield interleaved samples across all configured dataset sources."""
        generator = self.interleave_datasets(
            iterables=self.sources,
            weights=self.weights,
            seed=self.seed,
            stopping_strategy=self.stopping_strategy,
        )

        for sample in generator:
            if self.tokenizer is not None and isinstance(sample, dict) and "text" in sample:
                encoded = self.tokenizer(
                    sample["text"],
                    truncation=True,
                    max_length=self.max_length,
                )
                if isinstance(encoded, dict):
                    sample.update(encoded)
            yield sample


def create_multidataset_loader(
    dataset_names: Union[str, Sequence[Union[str, Any]]],
    subsets: Optional[Union[str, Sequence[Optional[str]], Dict[str, Optional[str]]]] = None,
    weights: Optional[Union[str, Sequence[float]]] = None,
    streaming: bool = True,
    split: str = "train",
    batch_size: Optional[int] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
    **kwargs: Any,
) -> Union[MultiDatasetLoader, DataLoader]:
    """Factory helper to construct a MultiDatasetLoader or PyTorch DataLoader."""
    loader = MultiDatasetLoader(
        dataset_names=dataset_names,
        subsets=subsets,
        weights=weights,
        streaming=streaming,
        split=split,
        seed=seed,
        stopping_strategy=stopping_strategy,
        **kwargs,
    )
    if batch_size is None:
        return loader
    return DataLoader(loader, batch_size=batch_size, collate_fn=collate_fn)


__all__ = [
    "MultiDatasetLoader",
    "create_multidataset_loader",
]
