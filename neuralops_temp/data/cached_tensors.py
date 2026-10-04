"""
Pre-tokenized Tensor Dataset Caching & Zero-Overhead In-Memory Loader.

Tokenizes dataset splits once (e.g. SST-2 for DistilBERT) and persists as .pt files.
Serves in-memory TensorDataset instances directly to persistent worker processes,
eliminating 20-30s of HuggingFace tokenization and multi-process DataLoader spawn
overhead per module evaluation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger("CachedTensors")

# Standard tokenizer mapping
MODEL_TOKENIZER_MAPPING = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "gpt2": "gpt2",
}

GLUE_TASK_CONFIG = {
    "sst2": {"keys": ("sentence", None), "num_labels": 2, "type": "classification"},
    "mrpc": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "type": "classification",
    },
    "qnli": {
        "keys": ("question", "sentence"),
        "num_labels": 2,
        "type": "classification",
    },
    "mnli": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "type": "classification",
    },
    "cola": {"keys": ("sentence", None), "num_labels": 2, "type": "classification"},
    "rte": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "type": "classification",
    },
    "stsb": {"keys": ("sentence1", "sentence2"), "num_labels": 1, "type": "regression"},
    "qqp": {
        "keys": ("question1", "question2"),
        "num_labels": 2,
        "type": "classification",
    },
}

_IN_MEMORY_CACHE: dict[tuple[str, str, int], dict[str, TensorDataset]] = {}


def get_cached_tokenized_dataset(
    dataset_name: str = "sst2",
    model_family: str = "distilbert",
    seq_len: int = 128,
    cache_dir: str | Path | None = None,
) -> dict[str, TensorDataset]:
    """Retrieves pre-tokenized TensorDatasets from RAM cache or disk cache (.pt).

    If missing on disk, automatically downloads, tokenizes, and saves the .pt cache.
    """
    key = (dataset_name.lower(), model_family.lower(), seq_len)
    if key in _IN_MEMORY_CACHE:
        return _IN_MEMORY_CACHE[key]

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "data_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{dataset_name.lower()}_{model_family.lower()}_len{seq_len}.pt"

    if cache_file.exists():
        t0 = time.time()
        logger.info(f"Loading pre-tokenized dataset cache from {cache_file}...")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        result = {
            "train": TensorDataset(
                data["train_input_ids"],
                data["train_attention_mask"],
                data["train_labels"],
            ),
            "val": TensorDataset(
                data["val_input_ids"],
                data["val_attention_mask"],
                data["val_labels"],
            ),
        }
        n_tr = len(result["train"])
        n_va = len(result["val"])
        logger.info(f"Loaded {n_tr} train and {n_va} val samples in {time.time() - t0:.3f}s")
        return result

    # Build cache
    logger.info(
        f"Pre-tokenized cache missing for {key}. Generating once and saving to {cache_file}..."
    )
    t0 = time.time()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    task_cfg = GLUE_TASK_CONFIG.get(
        dataset_name.lower(), {"keys": ("sentence", None), "num_labels": 2}
    )
    k1, k2 = task_cfg["keys"]

    def _tokenize_split(
        split_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_ds = load_dataset("glue", dataset_name.lower(), split=split_name)

        def tok_fn(ex):
            args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
            return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_len)

        tokenized = raw_ds.map(tok_fn, batched=True, desc=f"Tokenizing {dataset_name} {split_name}")

        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        label_key = "label" if "label" in tokenized.column_names else "labels"
        labels = torch.tensor(
            tokenized[label_key],
            dtype=torch.float if dataset_name.lower() == "stsb" else torch.long,
        )
        return input_ids, attention_mask, labels

    train_ids, train_mask, train_labels = _tokenize_split("train")
    val_ids, val_mask, val_labels = _tokenize_split("validation")

    save_dict = {
        "train_input_ids": train_ids,
        "train_attention_mask": train_mask,
        "train_labels": train_labels,
        "val_input_ids": val_ids,
        "val_attention_mask": val_mask,
        "val_labels": val_labels,
    }
    torch.save(save_dict, cache_file)
    logger.info(
        f"Generated and saved pre-tokenized cache ({cache_file.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.2f}s"
    )

    result = {
        "train": TensorDataset(train_ids, train_mask, train_labels),
        "val": TensorDataset(val_ids, val_mask, val_labels),
    }
    _IN_MEMORY_CACHE[key] = result
    return result
