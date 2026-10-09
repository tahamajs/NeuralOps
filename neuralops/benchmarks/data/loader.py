"""Universal Dataset Loader supporting ANY Hugging Face dataset, local files, and streaming."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from neuralops.benchmarks.results.schema import DatasetMetadata

LOG = logging.getLogger(__name__)

# Registry for custom dataset format loaders
DATASET_LOADER_REGISTRY: Dict[str, Type["DatasetLoader"]] = {}


def register_dataset_loader(format_name: str) -> Callable[[Type["DatasetLoader"]], Type["DatasetLoader"]]:
    """Decorator to register a custom dataset loader."""
    def decorator(cls: Type["DatasetLoader"]) -> Type["DatasetLoader"]:
        DATASET_LOADER_REGISTRY[format_name.lower()] = cls
        return cls
    return decorator


class DatasetLoader:
    """Base class for custom dataset loaders."""

    def load(self, path: Path | str, **kwargs: Any) -> Any:
        raise NotImplementedError


@register_dataset_loader("csv")
class CSVDatasetLoader(DatasetLoader):
    def load(self, path: Path | str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(path)


@register_dataset_loader("json")
@register_dataset_loader("jsonl")
class JSONDatasetLoader(DatasetLoader):
    def load(self, path: Path | str, **kwargs: Any) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".jsonl":
            return pd.read_json(p, lines=True)
        return pd.read_json(p)


@register_dataset_loader("parquet")
class ParquetDatasetLoader(DatasetLoader):
    def load(self, path: Path | str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(path)


def detect_dataset_task(column_names: List[str], dataset_name: str = "") -> str:
    """Auto-detect task type from column names and dataset identifier heuristics."""
    cols_lower = [c.lower() for c in column_names]
    name_lower = dataset_name.lower()

    if "instruction" in cols_lower and ("output" in cols_lower or "response" in cols_lower):
        return "instruction_tuning"
    if "question" in cols_lower and "context" in cols_lower:
        return "question_answering"
    if "document" in cols_lower and "summary" in cols_lower:
        return "summarization"
    if "translation" in cols_lower:
        return "translation"
    if any(k in name_lower for k in ("wikitext", "c4", "pile", "ptb")):
        return "language_modeling"
    if any(k in name_lower for k in ("arc", "hellaswag", "mmlu", "gsm8k")):
        return "multiple_choice"
    if "label" in cols_lower or "labels" in cols_lower or "target" in cols_lower or "sentiment" in cols_lower:
        return "classification"
    return "classification"


def auto_detect_columns(column_names: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Identify text, pair_text, and label column names from dataset headers."""
    cols = list(column_names)
    text_col = cols[0] if cols else "text"
    pair_col: Optional[str] = None
    label_col: Optional[str] = None

    # Detect text column
    for c in ("text", "sentence", "review", "prompt", "input", "document", "question"):
        if c in cols:
            text_col = c
            break

    # Detect second text column for pair tasks
    for c in ("sentence2", "hypothesis", "context"):
        if c in cols:
            pair_col = c
            break

    # Detect label column
    for c in ("label", "labels", "target", "sentiment", "score", "answer"):
        if c in cols:
            label_col = c
            break

    return text_col, pair_col, label_col


def load_universal_dataset(
    dataset_name_or_path: str,
    tokenizer: Any,
    task: str = "classification",
    split: str = "train",
    max_length: int = 128,
    max_samples: Optional[int] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    language: Optional[str] = None,
    streaming: bool = False,
) -> Tuple[Dataset, DatasetMetadata]:
    """
    Load ANY Hugging Face or local file dataset with automatic formatting:
    - Hugging Face datasets: 'sst2', 'glue/mrpc', 'squad', 'cnn_dailymail', 'tatsu-lab/alpaca', 'xnli'
    - Local formats: CSV (.csv), JSON/JSONL (.json, .jsonl), Parquet (.parquet), Text (.txt)
    """
    p = Path(dataset_name_or_path)
    is_local_file = p.exists() and p.is_file()

    texts: List[str] = []
    texts2: Optional[List[str]] = None
    labels: Optional[List[Any]] = None
    num_labels = 2

    # 1. Local File Loading
    if is_local_file:
        suffix = p.suffix.lower().replace(".", "")
        if suffix in DATASET_LOADER_REGISTRY:
            df = DATASET_LOADER_REGISTRY[suffix]().load(p)
        else:
            df = pd.read_csv(p)

        c_text, c_pair, c_label = auto_detect_columns(list(df.columns))
        t_col = text_column or c_text
        l_col = label_column or c_label

        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples]

        texts = [str(x) for x in df[t_col].tolist()]
        if c_pair and c_pair in df.columns:
            texts2 = [str(x) for x in df[c_pair].tolist()]
        if l_col and l_col in df.columns:
            raw_labels = df[l_col].tolist()
            # If string categories, convert to unique int IDs
            if raw_labels and isinstance(raw_labels[0], str):
                unique_cats = sorted(list(set(raw_labels)))
                cat_to_id = {c: i for i, c in enumerate(unique_cats)}
                labels = [cat_to_id[c] for c in raw_labels]
                num_labels = len(unique_cats)
            else:
                labels = [int(x) for x in raw_labels]
                num_labels = max(labels) + 1 if labels else 2

    # 2. Hugging Face Hub Dataset Loading
    else:
        from datasets import load_dataset
        ds_name = dataset_name_or_path
        hf_kwargs: Dict[str, Any] = {}
        if language:
            hf_kwargs["language"] = language

        # Handle subsets like glue/sst2 or wikitext
        if "/" in ds_name and not ds_name.startswith("tatsu-lab"):
            parts = ds_name.split("/", 1)
            raw_ds = load_dataset(parts[0], parts[1], split=split, streaming=streaming, **hf_kwargs)
        elif ds_name.lower() in ("sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "cola", "stsb", "wnli"):
            raw_ds = load_dataset("glue", ds_name.lower(), split=split, streaming=streaming, **hf_kwargs)
        else:
            raw_ds = load_dataset(ds_name, split=split, streaming=streaming, **hf_kwargs)

        # Handle streaming slice
        if streaming and max_samples:
            raw_ds = raw_ds.take(max_samples)

        c_text, c_pair, c_label = auto_detect_columns(raw_ds.column_names)
        t_col = text_column or c_text
        l_col = label_column or c_label

        limit = max_samples or (len(raw_ds) if hasattr(raw_ds, "__len__") else 1000)
        count = 0
        texts = []
        labels_list = []
        texts2_list = []

        for row in raw_ds:
            if count >= limit:
                break
            texts.append(str(row.get(t_col, "")))
            if c_pair and c_pair in row:
                texts2_list.append(str(row.get(c_pair, "")))
            if l_col and l_col in row and row[l_col] is not None:
                labels_list.append(row[l_col])
            count += 1

        if texts2_list:
            texts2 = texts2_list
        if labels_list:
            if isinstance(labels_list[0], str):
                unique_cats = sorted(list(set(labels_list)))
                cat_to_id = {c: i for i, c in enumerate(unique_cats)}
                labels = [cat_to_id[c] for c in labels_list]
                num_labels = len(unique_cats)
            else:
                labels = [int(x) for x in labels_list]
                num_labels = max(labels) + 1 if labels else 2

    # 3. Tokenize into TensorDataset
    if texts2 is not None:
        enc = tokenizer(
            texts,
            texts2,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None and len(labels) == len(texts):
        tensors.append(torch.tensor(labels, dtype=torch.long))
    else:
        # Default labels to input_ids for autoregressive tasks
        tensors.append(enc["input_ids"])

    if "token_type_ids" in enc:
        tensors.append(enc["token_type_ids"])

    tensor_dataset = TensorDataset(*tensors)

    metadata = DatasetMetadata(
        name=p.name if is_local_file else dataset_name_or_path,
        split=split,
        num_train=len(tensor_dataset) if split == "train" else 0,
        num_eval=len(tensor_dataset) if split in ("validation", "test", "val") else 0,
        num_labels=num_labels,
        task_type=task,
        max_seq_length=max_length,
        tokenizer=getattr(tokenizer, "name_or_path", ""),
        task=task,
    )
    return tensor_dataset, metadata


def make_universal_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """Wrap PyTorch Dataset in DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


__all__ = [
    "DatasetLoader",
    "register_dataset_loader",
    "detect_dataset_task",
    "auto_detect_columns",
    "load_universal_dataset",
    "make_universal_dataloader",
    "DATASET_LOADER_REGISTRY",
]
