"""Dataset loading, batch encoding, and dataloader construction for benchmarks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger(__name__)


def load_glue_dataset(
    task: str,
    tokenizer: Any,
    seq_len: int = 128,
    dataset_dir: Path | str | None = None,
    split: str = "validation",
) -> tuple[dict[str, torch.Tensor], int]:
    """
    Load and tokenize a standard GLUE dataset (SST-2, MRPC, QNLI, etc.).

    Args:
        task: GLUE task identifier (e.g. 'sst2', 'mrpc').
        tokenizer: PreTrainedTokenizer instance.
        seq_len: Maximum token sequence length.
        dataset_dir: Optional local directory containing cached Arrow datasets.
        split: Target split ('train', 'validation', 'test').

    Returns:
        Tuple of (batch_dict, num_labels).
    """
    task_lower = task.lower()
    from datasets import load_dataset

    LOG.info("Loading GLUE task '%s' (%s split)...", task_lower, split)
    if dataset_dir and Path(dataset_dir).exists():
        raw_ds = load_dataset(str(dataset_dir), split=split)
    else:
        raw_ds = load_dataset("glue", task_lower, split=split)

    field_map = {
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "cola": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
    }

    f1, f2 = field_map.get(task_lower, ("sentence", None))
    num_labels = 1 if task_lower == "stsb" else (3 if task_lower == "mnli" else 2)

    texts1 = raw_ds[f1]
    texts2 = raw_ds[f2] if f2 else None

    if texts2 is not None:
        encoded = tokenizer(
            texts1,
            texts2,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            texts1,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )

    labels = torch.tensor(raw_ds["label"])
    encoded["labels"] = labels
    return encoded, num_labels


def load_wikitext_dataset(
    tokenizer: Any,
    dataset_name: str = "wikitext-2-raw-v1",
    seq_len: int = 1024,
    split: str = "test",
) -> torch.Tensor:
    """
    Load Wikitext dataset and format as contiguous input IDs for causal LM evaluation.

    Args:
        tokenizer: PreTrainedTokenizer instance.
        dataset_name: HuggingFace dataset subset ('wikitext-2-raw-v1').
        seq_len: Maximum chunk sequence length.
        split: Target split ('test', 'validation').

    Returns:
        Tensor of shape [num_chunks, seq_len] with input token IDs.
    """
    from datasets import load_dataset

    LOG.info("Loading '%s' (%s split)...", dataset_name, split)
    ds = load_dataset("wikitext", dataset_name, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids[0]
    total_tokens = (input_ids.numel() // seq_len) * seq_len
    input_ids = input_ids[:total_tokens]
    chunks = input_ids.view(-1, seq_len)
    return chunks


def make_dataloader(
    encoded_batch: dict[str, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Create a standard PyTorch DataLoader from tokenized tensor dictionary."""
    tensors = [
        encoded_batch["input_ids"],
        encoded_batch["attention_mask"],
        encoded_batch["labels"],
    ]
    if "token_type_ids" in encoded_batch:
        tensors.append(encoded_batch["token_type_ids"])

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


__all__ = [
    "load_glue_dataset",
    "load_wikitext_dataset",
    "make_dataloader",
]
