"""Evaluation data loading, batch generation, and calibration loader builder.

Handles tokenized caching, real GLUE/LM dataset batch retrieval, synthetic data fallbacks,
and real calibration DataLoaders for activation-aware model compression.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuralops.evaluation.eval_types import (
    GLUE_DATASETS,
    LM_DATASETS,
    MODEL_TOKENIZER_MAPPING,
)
from neuralops.evaluation.eval_utils import _evict_if_needed

logger = logging.getLogger("EvalEngine")

# Caches for tokenized evaluation batches and loaded huggingface datasets
_TOKENIZED_BATCH_CACHE: dict[tuple[str, str, int, int, str], dict[str, torch.Tensor]] = {}
_PREPARED_DATASET_CACHE: dict[tuple[str, str, int], Any] = {}


def _generate_synthetic_batch(
    dataset_name: str,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Generate reproducible synthetic tensor batches for smoke tests or offline environments.

    Args:
        dataset_name: Dataset name identifier (used to check if LM task).
        batch_size: Number of samples in the batch.
        seq_len: Sequence length dimension.
        vocab_size: Maximum vocabulary size for random token IDs.
        num_labels: Number of target classification classes.
        device: PyTorch device on which tensors will be allocated.

    Returns:
        Dictionary mapping input names ('input_ids', 'attention_mask', 'labels') to tensors.
    """
    ds = dataset_name.lower()
    torch.manual_seed(42)
    input_ids = torch.randint(100, max(101, vocab_size - 100), (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    is_lm = ds in LM_DATASETS
    if batch_size > 1 and not is_lm:
        attention_mask[1:, -4:] = 0

    if is_lm:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
    else:
        labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_dataset_batch(
    dataset_name: str,
    model_family: str,
    batch_size: int = 128,
    seq_len: int = 256,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> dict[str, torch.Tensor]:
    """Retrieve an evaluation batch from cache, genuine dataset (GLUE / WikiText), or synthetic fallback.

    Args:
        dataset_name: Target dataset (e.g., 'sst2', 'wikitext2', 'synthetic').
        model_family: Target model family for matching the tokenizer.
        batch_size: Number of samples requested.
        seq_len: Target sequence length for truncation/padding.
        vocab_size: Vocab size for synthetic fallback.
        num_labels: Classification label count.
        device: PyTorch device to move batch tensors to.
        allow_synthetic: If False, raises RuntimeError if genuine dataset fails to download.

    Returns:
        Dictionary mapping tensor keys to torch.Tensor.
    """
    ds = dataset_name.lower()
    family = model_family.lower()
    dev_str = str(device)
    cache_key = (ds, family, batch_size, seq_len, dev_str)

    if cache_key in _TOKENIZED_BATCH_CACHE:
        cached = _TOKENIZED_BATCH_CACHE[cache_key]
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cached.items()}

    if ds == "synthetic":
        batch = _generate_synthetic_batch(ds, batch_size, seq_len, vocab_size, num_labels, device)
        _TOKENIZED_BATCH_CACHE[cache_key] = batch
        _evict_if_needed(_TOKENIZED_BATCH_CACHE)
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    tok_name = MODEL_TOKENIZER_MAPPING.get(family, "bert-base-uncased")
    dataset_cache_key = (ds, tok_name, seq_len)

    try:
        from neuralops.data.prepare_datasets import GLUE_TASK_CONFIG

        val_tokenized = None
        if ds in GLUE_TASK_CONFIG:
            if dataset_cache_key not in _PREPARED_DATASET_CACHE:
                from datasets import load_dataset
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tok_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                config = GLUE_TASK_CONFIG[ds]
                k1, k2 = config["keys"]
                val_data = load_dataset("glue", ds, split="validation")

                def tok_fn(ex: dict[str, Any]) -> Any:
                    """Execute tok_fn operation."""
                    args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
                    return tokenizer(
                        *args, padding="max_length", truncation=True, max_length=seq_len
                    )

                val_tokenized = val_data.map(tok_fn, batched=True)
                val_tokenized.set_format(
                    type="torch",
                    columns=[
                        c
                        for c in ("input_ids", "attention_mask", "label")
                        if c in val_tokenized.column_names
                    ],
                )
                _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
                _evict_if_needed(_PREPARED_DATASET_CACHE)
            else:
                val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]
        elif ds in LM_DATASETS or "wikitext" in ds:
            if dataset_cache_key not in _PREPARED_DATASET_CACHE:
                from datasets import Dataset, load_dataset
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tok_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                try:
                    raw_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
                except Exception:
                    raw_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

                valid_texts = [t for t in raw_val["text"] if len(t.strip()) > 10][:200]
                if not valid_texts:
                    valid_texts = [
                        "WikiText language model benchmark sample evaluation string."
                    ] * 50

                enc = tokenizer(
                    valid_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=seq_len,
                    return_tensors="pt",
                )
                val_tokenized = Dataset.from_dict(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                    }
                )
                val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
                _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
                _evict_if_needed(_PREPARED_DATASET_CACHE)
            else:
                val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]

        if val_tokenized is not None and len(val_tokenized) > 0:
            is_cuda = device.type == "cuda"
            n = min(batch_size, len(val_tokenized))
            sample = val_tokenized[list(range(n))]
            raw_batch = {}
            for k in val_tokenized.column_names:
                if k in sample:
                    vals = sample[k]
                    if isinstance(vals, torch.Tensor):
                        raw_batch[k] = vals
                    elif (
                        isinstance(vals, (list, tuple))
                        and len(vals) > 0
                        and isinstance(vals[0], torch.Tensor)
                    ):
                        raw_batch[k] = torch.stack(vals)
                    elif (
                        isinstance(vals, (list, tuple))
                        and len(vals) > 0
                        and isinstance(vals[0], (int, float, bool))
                    ):
                        raw_batch[k] = torch.tensor(vals)
                    elif isinstance(vals, (int, float, bool)):
                        raw_batch[k] = torch.tensor([vals])
                    else:
                        raw_batch[k] = vals
            batch = {
                ("labels" if k == "label" else k): (
                    v.to(device, non_blocking=is_cuda) if isinstance(v, torch.Tensor) else v
                )
                for k, v in raw_batch.items()
            }
            _TOKENIZED_BATCH_CACHE[cache_key] = batch
            _evict_if_needed(_TOKENIZED_BATCH_CACHE)
            return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    except Exception as e:
        logger.warning("Could not load GLUE dataset '%s': %s", ds, e)
        if not allow_synthetic:
            raise RuntimeError(f"Genuine dataset '{ds}' failed to load ({e})")

    batch = _generate_synthetic_batch(ds, batch_size, seq_len, vocab_size, num_labels, device)
    _TOKENIZED_BATCH_CACHE[cache_key] = batch
    _evict_if_needed(_TOKENIZED_BATCH_CACHE)
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def build_calibration_loader(
    dataset: str,
    model_family: str,
    seq_len: int,
    device: torch.device,
    n_samples: int = 500,
    batch_size: int = 8,
) -> tuple[DataLoader | None, Any | None, str | None]:
    """Build a calibration DataLoader from the training split for activation-aware compressors.

    Args:
        dataset: Dataset identifier string (GLUE or LM).
        model_family: Architecture family to match tokenizer.
        seq_len: Token sequence length.
        device: Device to calibrate on.
        n_samples: Maximum number of samples to tokenize for calibration.
        batch_size: Batch size of the calibration loader.

    Returns:
        Tuple of (DataLoader, AutoTokenizer, source_name_str), or (None, None, None) on failure.
    """
    from transformers import AutoTokenizer

    tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    except Exception as e:
        logger.warning("[CALIB] tokenizer load failed: %s", e)
        return None, None, None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.pad_token

    ds = dataset.lower()
    source: str | None = None
    texts: list[str] = []
    try:
        from datasets import load_dataset

        if ds in GLUE_DATASETS:
            raw = load_dataset("glue", ds, split=f"train[:{n_samples}]")
            text_cols = [c for c in raw.column_names if c not in ("label", "idx")]
            if not text_cols:
                return None, None, None
            col = text_cols[0]
            texts = raw[col][:n_samples]
            source = f"glue/{ds}_train"
        elif ds in LM_DATASETS:
            raw = load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1", split=f"train[:{n_samples}]"
            )
            texts = [t for t in raw["text"] if len(t.strip()) > 20][:n_samples]
            source = "wikitext2_train"
        else:
            return None, None, None
    except Exception as e:
        logger.warning("[CALIB] dataset load failed: %s", e)
        return None, None, None

    if not texts:
        return None, None, None

    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
        for i in range(enc["input_ids"].shape[0])
    ]

    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    logger.info(
        "[CALIB] Built calibration loader from %s: n=%d, batch_size=%d, seq_len=%d",
        source,
        len(items),
        batch_size,
        seq_len,
    )
    return loader, tokenizer, source


__all__ = [
    "get_dataset_batch",
    "build_calibration_loader",
    "_generate_synthetic_batch",
]
