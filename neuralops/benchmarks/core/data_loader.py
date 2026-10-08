"""Dataset loading, batch formatting, and DataLoader creation for benchmarks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from neuralops.benchmarks.result import DatasetMetadata

LOG = logging.getLogger(__name__)

GLUE_TASKS: dict[str, dict[str, Any]] = {
    "sst2": {"cols": ("sentence", None), "num_labels": 2, "task": "classification"},
    "mrpc": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification"},
    "qqp": {"cols": ("question1", "question2"), "num_labels": 2, "task": "classification"},
    "mnli": {"cols": ("premise", "hypothesis"), "num_labels": 3, "task": "classification"},
    "qnli": {"cols": ("question", "sentence"), "num_labels": 2, "task": "classification"},
    "rte": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification"},
    "cola": {"cols": ("sentence", None), "num_labels": 2, "task": "classification"},
    "stsb": {"cols": ("sentence1", "sentence2"), "num_labels": 1, "task": "regression"},
    "wnli": {"cols": ("sentence1", "sentence2"), "num_labels": 2, "task": "classification"},
}


def get_num_labels(dataset_name: str) -> int:
    """Return number of classification labels for dataset."""
    name_clean = dataset_name.lower().replace("glue_", "")
    if name_clean in GLUE_TASKS:
        return GLUE_TASKS[name_clean]["num_labels"]
    return 2


def get_dataset_metadata(dataset: Any, name: str) -> DatasetMetadata:
    """Construct standard DatasetMetadata instance."""
    name_clean = name.lower().replace("glue_", "")
    info = GLUE_TASKS.get(name_clean, {"num_labels": 2, "task": "classification"})
    total_samples = len(dataset) if hasattr(dataset, "__len__") else 0
    return DatasetMetadata(
        name=name,
        task=info.get("task", "classification"),
        num_labels=info.get("num_labels", 2),
        num_train=total_samples,
        num_eval=total_samples,
    )


def load_dataset_by_name(
    name: str,
    tokenizer: Any,
    max_length: int = 128,
    split: str = "validation",
) -> Dataset:
    """
    Load dataset by name from local cache or HuggingFace hub.
    Supports GLUE tasks and language modeling (Wikitext).
    """
    name_clean = name.lower().replace("glue_", "")
    from datasets import load_dataset

    # 1. GLUE Classification
    if name_clean in GLUE_TASKS:
        cfg = GLUE_TASKS[name_clean]
        raw_ds = load_dataset("glue", name_clean, split=split)
        c1, c2 = cfg["cols"]

        texts1 = raw_ds[c1]
        texts2 = raw_ds[c2] if c2 else None

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

    # 2. Wikitext Language Modeling
    if "wikitext" in name_clean:
        subset = "wikitext-2-raw-v1" if "wikitext2" in name_clean or "wikitext-2" in name_clean else "wikitext-103-raw-v1"
        split_name = "test" if split in ("val", "validation", "test") else split
        raw_ds = load_dataset("wikitext", subset, split=split_name)
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
) -> DataLoader:
    """Wrap standard Dataset in PyTorch DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

