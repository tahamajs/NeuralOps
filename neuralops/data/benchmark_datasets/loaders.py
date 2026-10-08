"""Benchmark dataset loader and PyTorch dataset wrapping.

Implements BenchmarkTorchDataset for tokenized PyTorch DataLoader creation,
the unified load_benchmark_dataset entry point with streaming and fallback logic,
and get_synthetic_benchmark_sample.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neuralops.data.benchmark_datasets.manifest import (
    BenchmarkDatasetInfo,
    BenchmarkDatasetRegistry,
)

logger = logging.getLogger(__name__)


class BenchmarkTorchDataset(Dataset):
    """PyTorch Dataset wrapper for processed benchmark records."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any = None,
        seq_len: int = 2048,
        is_causal_lm: bool = True,
    ):
        """Initialize BenchmarkTorchDataset with sample dictionary list.

        Args:
            samples: List of raw sample dictionaries.
            tokenizer: Optional HuggingFace tokenizer.
            seq_len: Maximum token sequence length.
            is_causal_lm: True if sequence is for causal language modeling.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.is_causal_lm = is_causal_lm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        if self.tokenizer is None:
            return item

        text = item.get("text", "")
        if not text:
            for k in ["prompt", "sentence", "question", "sentence1"]:
                if k in item and item[k]:
                    text = str(item[k])
                    break

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "label" in item:
            data["labels"] = torch.tensor(
                item["label"],
                dtype=torch.long if isinstance(item["label"], int) else torch.float,
            )
        elif self.is_causal_lm:
            labels = input_ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return data


def load_benchmark_dataset(
    dataset_name: str,
    tokenizer: Any = None,
    seq_len: int = 2048,
    batch_size: int = 8,
    split: str | None = None,
    max_samples: int = 1000,
    streaming: bool = True,
    cache_dir: str | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    use_synthetic: bool = False,
    seed: int = 42,
    return_dataloader: bool = True,
    **kwargs: Any,
) -> DataLoader | list[dict[str, Any]]:
    """Unified loader for 12+ NLP / LLM benchmark datasets.

    Supports:
      - wikitext2 / wikitext103
      - c4 / c4_en
      - ptb
      - pg19
      - gsm8k
      - humaneval
      - mmlu
      - pile / pile_val
      - arc (ARC-Easy & ARC-Challenge)
      - hellaswag
      - lambada
      - glue (sst2, mrpc, qqp, mnli, qnli, rte, wnli, cola, stsb)

    Args:
        dataset_name: Benchmark name or alias (e.g. 'wikitext2', 'c4', 'gsm8k', 'arc', 'glue/sst2').
        tokenizer: HuggingFace tokenizer instance (optional).
        seq_len: Maximum sequence length for tokenization.
        batch_size: DataLoader batch size.
        split: Dataset split ('train', 'validation', 'test'). Defaults to benchmark standard.
        max_samples: Maximum number of samples to load/stream.
        streaming: Whether to stream from HuggingFace Hub to avoid large downloads.
        cache_dir: Local directory to cache datasets.
        shuffle: Whether to shuffle samples.
        num_workers: Number of workers for DataLoader.
        use_synthetic: Force synthetic data generation for offline testing.
        seed: Random seed for reproducibility.
        return_dataloader: If True, returns PyTorch DataLoader; otherwise returns list of dicts.
        **kwargs: Additional arguments passed to datasets.load_dataset.

    Returns:
        torch.utils.data.DataLoader or List[Dict[str, Any]].
    """
    clean_name = dataset_name.lower().replace("-", "_")
    info = BenchmarkDatasetRegistry.get(clean_name)

    if info is None and "/" in dataset_name:
        prefix, subtask = dataset_name.split("/", 1)
        if prefix.lower() == "glue":
            info = BenchmarkDatasetRegistry.get(f"glue_{subtask}")

    if info is None:
        logger.warning(
            "Dataset '%s' not in official benchmark registry. Creating generic fallback.",
            dataset_name,
        )
        info = BenchmarkDatasetInfo(
            name=clean_name,
            aliases=[],
            category="custom",
            default_hf_path=dataset_name,
            default_split=split or "validation",
            text_formatter=lambda r: {"text": r.get("text", str(r))},
            synthetic_generator=lambda n, s: [
                {
                    "text": f"Synthetic custom text sample #{i} for {dataset_name}.",
                    "id": i,
                }
                for i in range(n)
            ],
        )

    target_split = split or info.default_split
    raw_samples: list[dict[str, Any]] = []

    if not use_synthetic:
        try:
            from datasets import load_dataset as hf_load_dataset

            hf_kwargs = dict(kwargs)
            if info.default_subset:
                args = (info.default_hf_path, info.default_subset)
            else:
                args = (info.default_hf_path,)

            if cache_dir:
                hf_kwargs["cache_dir"] = cache_dir

            logger.info("Attempting to load benchmark '%s' from HF (%s)...", info.name, args)

            try:
                if streaming:
                    hf_ds = hf_load_dataset(*args, split=target_split, streaming=True, **hf_kwargs)
                    count = 0
                    for row in hf_ds:
                        formatted = info.text_formatter(row) if info.text_formatter else dict(row)
                        if (
                            formatted.get("text", "").strip()
                            or "prompt" in formatted
                            or "question" in formatted
                        ):
                            raw_samples.append(formatted)
                            count += 1
                            if count >= max_samples:
                                break
                else:
                    hf_ds = hf_load_dataset(*args, split=target_split, **hf_kwargs)
                    sub_ds = hf_ds.select(range(min(len(hf_ds), max_samples)))
                    for row in sub_ds:
                        formatted = info.text_formatter(row) if info.text_formatter else dict(row)
                        raw_samples.append(formatted)

            except Exception as inner_e:
                logger.warning(
                    "Streaming/split load failed for %s (%s). Retrying non-streaming or default split...",
                    info.name,
                    inner_e,
                )
                hf_ds = hf_load_dataset(*args, **hf_kwargs)
                actual_split = target_split if target_split in hf_ds else list(hf_ds.keys())[0]
                ds_split = hf_ds[actual_split]
                sub_ds = ds_split.select(range(min(len(ds_split), max_samples)))
                for row in sub_ds:
                    formatted = info.text_formatter(row) if info.text_formatter else dict(row)
                    raw_samples.append(formatted)

        except Exception as e:
            logger.warning(
                "Failed to load '%s' from HuggingFace Hub: %s. Falling back to synthetic generator.",
                dataset_name,
                str(e),
            )
            raw_samples = []

    if not raw_samples:
        logger.info(
            "Generating %d synthetic samples for benchmark '%s' (seed=%d)...",
            max_samples,
            info.name,
            seed,
        )
        if info.synthetic_generator:
            raw_samples = info.synthetic_generator(max_samples, seed)
        else:
            raw_samples = [
                {"text": f"Synthetic benchmark text #{i} for {info.name}", "id": i}
                for i in range(max_samples)
            ]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(raw_samples)

    if not return_dataloader:
        return raw_samples

    is_lm = info.category in ("language_modeling", "code")
    torch_ds = BenchmarkTorchDataset(
        samples=raw_samples,
        tokenizer=tokenizer,
        seq_len=seq_len,
        is_causal_lm=is_lm,
    )

    loader = DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def get_synthetic_benchmark_sample(
    dataset_name: str, num_samples: int = 4, seed: int = 42
) -> list[str]:
    """Retrieve raw text strings of synthetic benchmark samples.

    Args:
        dataset_name: Target benchmark name.
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        List of synthetic string samples.
    """
    raw = load_benchmark_dataset(
        dataset_name,
        max_samples=num_samples,
        use_synthetic=True,
        seed=seed,
        return_dataloader=False,
    )
    results = []
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text") or item.get("prompt") or item.get("sentence") or str(item)
            results.append(str(text))
        else:
            results.append(str(item))
    return results


__all__ = [
    "BenchmarkTorchDataset",
    "load_benchmark_dataset",
    "get_synthetic_benchmark_sample",
]
