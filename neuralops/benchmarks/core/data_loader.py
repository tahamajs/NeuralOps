#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dataset loading, batch formatting, and DataLoader creation for benchmarks

Implements data loader functionality for NeuralOps.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset


from neuralops.benchmarks.result import DatasetMetadata

LOG = logging.getLogger(__name__)

GLUE_TASKS: dict[str, dict[str, Any]] = {
    "sst2": {"cols": ("sentence", None), "num_labels": 2, "task": "classification", "val_size": 872},
    "mrpc": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification", "val_size": 408},
    "qqp": {"cols": ("question1", "question2"), "num_labels": 2, "task": "classification", "val_size": 40430},
    "mnli": {"cols": ("premise", "hypothesis"), "num_labels": 3, "task": "classification", "val_size": 9815},
    "qnli": {"cols": ("question", "sentence"), "num_labels": 2, "task": "classification", "val_size": 5463},
    "rte": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification", "val_size": 277},
    "cola": {"cols": ("sentence", None), "num_labels": 2, "task": "classification", "val_size": 1043},
    "stsb": {"cols": ("sentence1", "sentence2"), "num_labels": 1, "task": "regression", "val_size": 1500},
    "wnli": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification", "val_size": 71},
}

GLUE_VALIDATION_SIZES: dict[str, int] = {
    task: info["val_size"] for task, info in GLUE_TASKS.items()
}


class _TokenizedIterableDataset(IterableDataset):
    """Lazily tokenize a Hugging Face or local streaming dataset."""

    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer: Any, columns: tuple[str, str | None], max_length: int) -> None:
        super().__init__()
        self.rows = rows
        self.tokenizer = tokenizer
        self.columns = columns
        self.max_length = max_length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        first, second = self.columns
        for row in self.rows:
            args = (row[first], row[second]) if second else (row[first],)
            encoded = self.tokenizer(
                *args,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            encoded["labels"] = row.get("label", row.get("labels", 0))
            yield encoded


def _load_local_dataset(path: Path, streaming: bool, cache_dir: str | None = None) -> Any:
    """Load CSV/JSON/JSONL through ``datasets`` while preserving streaming."""
    from datasets import load_dataset

    suffix = path.suffix.lower()
    if suffix == ".csv":
        builder = "csv"
    elif suffix in {".json", ".jsonl"}:
        builder = "json"
    else:
        raise ValueError(f"Unsupported local dataset format '{suffix}'. Use .csv, .json, or .jsonl.")
    kwargs: dict[str, Any] = {"data_files": str(path), "split": "train", "streaming": streaming}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(builder, **kwargs)


def get_num_labels(dataset_name: str) -> int:
    """Return number of classification labels for dataset."""
    name_clean = dataset_name.lower().replace("glue_", "")
    if name_clean in GLUE_TASKS:
        return GLUE_TASKS[name_clean]["num_labels"]
    return 2


def get_dataset_metadata(
    dataset: Any,
    name: str,
    split: str | None = None,
    num_train: int | None = None,
    num_eval: int | None = None,
) -> DatasetMetadata:
    """Construct standard DatasetMetadata instance."""
    name_clean = name.lower().replace("glue_", "")
    info = GLUE_TASKS.get(name_clean, {"num_labels": 2, "task": "classification"})
    total_samples = len(dataset) if hasattr(dataset, "__len__") else 0

    expected_eval = num_eval
    if expected_eval is None and name_clean in GLUE_VALIDATION_SIZES:
        expected_eval = GLUE_VALIDATION_SIZES[name_clean]
    elif expected_eval is None:
        expected_eval = total_samples

    return DatasetMetadata(
        name=name,
        task=info.get("task", "classification"),
        num_labels=info.get("num_labels", 2),
        num_train=num_train if num_train is not None else total_samples,
        num_eval=expected_eval,
    )


def load_dataset_by_name(
    name: str,
    tokenizer: Any,
    max_length: int = 128,
    split: str = "validation",
    max_samples: int | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | IterableDataset:
    """
    Load GLUE, WikiText, or a local CSV/JSONL dataset.

    ``streaming=True`` returns a lazy iterable with tokenization performed as
    samples are consumed. Network/cache failures are converted into a clear
    ``RuntimeError``; callers may opt into synthetic data explicitly through
    :mod:`neuralops.evaluation.eval_data`.
    """
    import os
    hf_cache_dir = os.environ.get("HF_DATASETS_CACHE") or os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.isdir(hf_cache_dir):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if max_samples is None:
        env_samples = os.environ.get("NEURALOPS_MAX_SAMPLES")
        if env_samples and env_samples.isdigit() and int(env_samples) > 0:
            max_samples = int(env_samples)

    name_clean = name.lower().replace("glue_", "")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for GLUE, WikiText, and local dataset loading."
        ) from exc

    local_path = Path(name).expanduser()
    if local_path.is_file():
        try:
            raw_local = _load_local_dataset(local_path, streaming=streaming, cache_dir=hf_cache_dir)
            if streaming:
                columns = tuple(c for c in ("sentence", "text", "sentence1") if c in raw_local.column_names)
                first = columns[0] if columns else next(c for c in raw_local.column_names if c not in {"label", "labels"})
                second = "sentence2" if "sentence2" in raw_local.column_names else None
                return _TokenizedIterableDataset(raw_local, tokenizer, (first, second), max_length)
            rows = raw_local
            text_columns = [c for c in ("sentence", "text", "sentence1") if c in rows.column_names]
            first = text_columns[0] if text_columns else next(c for c in rows.column_names if c not in {"label", "labels"})
            second = "sentence2" if "sentence2" in rows.column_names else None
            texts1 = list(rows[first])
            texts2 = list(rows[second]) if second else None
            enc = tokenizer(
                texts1,
                texts2,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            labels = torch.tensor(
                rows["label"] if "label" in rows.column_names
                else rows["labels"] if "labels" in rows.column_names
                else [0] * len(rows)
            )
            return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local dataset '{local_path}': {exc}") from exc

    if name_clean in GLUE_TASKS:
        cfg = GLUE_TASKS[name_clean]
        try:
            raw_ds = load_dataset("nyu-mll/glue", name_clean, split=split, streaming=streaming, cache_dir=cache_dir)
        except Exception:
            try:
                raw_ds = load_dataset("glue", name_clean, split=split, streaming=streaming, cache_dir=cache_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load GLUE dataset '{name_clean}' split '{split}'. "
                    "Check the local cache or unset HF_DATASETS_OFFLINE."
                ) from exc
        if streaming:
            return _TokenizedIterableDataset(raw_ds, tokenizer, cfg["cols"], max_length)
        if max_samples is not None and max_samples < len(raw_ds):
            raw_ds = raw_ds.select(range(max_samples))
            LOG.info("Subsampled %s split '%s' to %d samples", name_clean, split, len(raw_ds))
        c1, c2 = cfg["cols"]

        texts1 = list(raw_ds[c1])
        texts2 = list(raw_ds[c2]) if c2 else None

        if texts2 is not None:
            enc = tokenizer(
                texts1,
                texts2,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            enc = tokenizer(
                texts1,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        labels = torch.tensor(raw_ds["label"])
        tensors = [enc["input_ids"], enc["attention_mask"], labels]
        if "token_type_ids" in enc:
            tensors.append(enc["token_type_ids"])
        return TensorDataset(*tensors)

    if "wikitext" in name_clean:
        subset = "wikitext-2-raw-v1" if "wikitext2" in name_clean or "wikitext-2" in name_clean else "wikitext-103-raw-v1"
        split_name = "test" if split in ("val", "validation", "test") else split
        try:
            raw_ds = load_dataset("Salesforce/wikitext", subset, split=split_name, streaming=streaming, cache_dir=cache_dir)
        except Exception:
            try:
                raw_ds = load_dataset("wikitext", subset, split=split_name, streaming=streaming, cache_dir=cache_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load WikiText dataset '{subset}' split '{split_name}'. "
                    "Check the local cache or unset HF_DATASETS_OFFLINE."
                ) from exc
        if streaming:
            return _TokenizedIterableDataset(raw_ds, tokenizer, ("text", None), max_length)
        if max_samples is not None and max_samples < len(raw_ds):
            raw_ds = raw_ds.select(range(max_samples))
            LOG.info("Subsampled WikiText split '%s' to %d samples", split_name, len(raw_ds))
        full_text = "\n\n".join(raw_ds["text"])
        enc = tokenizer(full_text, return_tensors="pt")
        input_ids = enc.input_ids[0]

        total_tokens = (input_ids.numel() // max_length) * max_length
        input_ids = input_ids[:total_tokens]
        chunks = input_ids.view(-1, max_length)
        attention_mask = torch.ones_like(chunks)
        return TensorDataset(chunks, attention_mask, chunks)

    raise ValueError(f"Unsupported benchmark dataset '{name}'. Available: {list(GLUE_TASKS.keys())} + wikitext2/103")


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int | None = None,
) -> DataLoader:
    """Create a DataLoader tuned for GPU training.

    On Linux, num_workers=4 allows parallel data prep while the GPU
    is computing. pin_memory=True enables DMA transfers.
    """
    if num_workers is None:
        num_workers = int(os.environ.get("NEURALOPS_DATALOADER_WORKERS", "0"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def load_glue_dataset(task_name: str, tokenizer: Any, max_length: int = 128, split: str = "validation") -> Dataset:
    """Backward-compatible alias for load_dataset_by_name on GLUE tasks."""
    return load_dataset_by_name(name=task_name, tokenizer=tokenizer, max_length=max_length, split=split)


def load_wikitext_dataset(tokenizer: Any, max_length: int = 512, split: str = "test", subset: str = "wikitext-2-raw-v1") -> Dataset:
    """Backward-compatible alias for load_dataset_by_name on wikitext."""
    ds_name = "wikitext2" if "wikitext-2" in subset else "wikitext103"
    return load_dataset_by_name(name=ds_name, tokenizer=tokenizer, max_length=max_length, split=split)


__all__ = [
    "GLUE_TASKS",
    "get_num_labels",
    "get_dataset_metadata",
    "load_dataset_by_name",
    "load_glue_dataset",
    "load_wikitext_dataset",
    "make_dataloader",
]
