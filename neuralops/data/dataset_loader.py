"""
Unified dataset loader driven by an ExperimentConfig.
Supports GLUE, custom CSV/JSON/Parquet, multi-choice, synthetic, and any HuggingFace hub dataset.
Equipped with streaming, local caching, tokenization, dynamic batch collating, and offline fallbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader

try:
    from datasets import Dataset, DatasetDict, IterableDataset, load_dataset as hf_load_dataset
except (ImportError, AttributeError):
    DatasetDict = dict
    Dataset = Any
    IterableDataset = Any
    hf_load_dataset = None

logger = logging.getLogger("NeuralOps.DatasetLoader")


class ExperimentConfig:
    """
    Configuration for loading a dataset.

    Attributes:
        dataset_name: Name of the dataset (GLUE task, synthetic, file path, or HF identifier).
        dataset_path: Path to local dataset file (for custom or multi-choice).
        task_type: "single", "pair", "multi_choice", "regression", or "language_modeling".
        text_columns: List of column names containing text (auto-detected if None).
        label_column: Name of the label column (auto-detected if None).
        num_labels: Number of classes (for classification).
        metric: Primary metric for evaluation (e.g., "accuracy").
        max_seq_length: Maximum sequence length for tokenization.
        split_ratio: Fraction of data to use for validation (for custom datasets).
        synthetic_samples: Number of synthetic samples to generate (if dataset_name="synthetic").
        glue_base_path: Base directory for local GLUE Parquet files.
        streaming: If True, stream dataset from HuggingFace without full download.
        cache_dir: Optional custom local directory for HF caching.
        subset: Optional dataset subset/configuration name (e.g., 'wikitext-2-raw-v1').
        split: Optional specific split to load (e.g., 'train', 'validation', 'test').
        **kwargs: Additional arguments passed to underlying loaders.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None = None,
        task_type: str = "single",
        text_columns: list[str] | None = None,
        label_column: str | None = None,
        num_labels: int = 2,
        metric: str = "accuracy",
        max_seq_length: int = 128,
        split_ratio: float = 0.1,
        synthetic_samples: int = 5000,
        glue_base_path: str | None = None,
        streaming: bool = False,
        cache_dir: str | None = None,
        subset: str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.text_columns = text_columns
        self.label_column = label_column
        self.num_labels = num_labels
        self.metric = metric
        self.max_seq_length = max_seq_length
        self.split_ratio = split_ratio
        self.synthetic_samples = synthetic_samples
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.subset = subset
        self.split = split

        from neuralops.utils.paths import get_glue_dir

        self.glue_base_path = glue_base_path or str(get_glue_dir())
        self.kwargs = kwargs


def load_dataset(config: ExperimentConfig) -> Any:
    """
    Load a dataset based on ExperimentConfig.
    Returns DatasetDict (or IterableDataset when streaming) with at least 'train'
    and optionally 'validation'/'test' splits.
    """
    name = config.dataset_name.lower()

    # GLUE tasks (local Parquet if available, or nyu-mll/glue HF hub fallback)
    if name in {"cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"}:
        return _load_glue(config)

    # Synthetic data generation
    elif name == "synthetic":
        return _load_synthetic(config)

    # Custom local file (CSV, JSON, Parquet)
    elif name.endswith((".csv", ".json", ".jsonl", ".parquet")):
        return _load_custom(config)

    # Multi-choice dataset (ARC, HellaSwag, etc.)
    elif name == "multi_choice":
        return _load_multi_choice(config)

    # Fallback: General HuggingFace dataset hub (e.g. wikitext, c4, openwebtext, ptb, user hub datasets)
    else:
        return _load_hub(config)


def _load_glue(config: ExperimentConfig) -> DatasetDict:
    """Load GLUE benchmark with local Parquet priority and seamless HF hub fallback."""
    from .glue_loader import load_glue_local

    try:
        return load_glue_local(config.dataset_name, config.glue_base_path)
    except Exception as e:
        logger.warning("Local GLUE parquet load failed (%s), falling back to HuggingFace Hub...", e)
        if hf_load_dataset is None:
            raise
        try:
            return hf_load_dataset("nyu-mll/glue", config.dataset_name.lower(), cache_dir=config.cache_dir)
        except Exception:
            return hf_load_dataset("glue", config.dataset_name.lower(), cache_dir=config.cache_dir)


def _load_synthetic(config: ExperimentConfig) -> DatasetDict:
    """Generate synthetic dataset in memory."""
    import tempfile
    from .synthetic_generator import run_universal_generation

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_universal_generation(
            output_dir=Path(tmpdir), total_samples_per_domain=config.synthetic_samples
        )
        dataset = Dataset.from_pandas(df)
        splits = dataset.train_test_split(test_size=config.split_ratio)
        return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def _load_custom(config: ExperimentConfig) -> DatasetDict:
    """Load custom tabular or text dataset from file."""
    from .custom_loader import load_dataset_from_file

    file_path = config.dataset_path or config.dataset_name
    text_col = config.text_columns[0] if config.text_columns else None
    return load_dataset_from_file(
        file_path=file_path,
        split_ratio=config.split_ratio,
        text_column=text_col,
        label_column=config.label_column,
        **config.kwargs,
    )


def _load_multi_choice(config: ExperimentConfig) -> DatasetDict:
    """Load multi-choice dataset format."""
    from .custom_loader import load_custom_dataset

    return load_custom_dataset(config.dataset_path or config.dataset_name)


def _load_hub(config: ExperimentConfig) -> Any:
    """
    Load any HuggingFace Hub dataset with support for:
    - Streaming (config.streaming=True)
    - Well-known aliases (e.g. wikitext-2, c4, openwebtext, ptb)
    - Local directory caching
    - Subsets and specific splits
    - Automatic train/validation split creation if validation split is absent
    """
    if hf_load_dataset is None:
        raise ImportError("HuggingFace 'datasets' library is required to load from Hub.")

    name = config.dataset_name
    name_lower = name.lower()
    load_kwargs = dict(config.kwargs)

    if config.cache_dir:
        load_kwargs["cache_dir"] = config.cache_dir
    if config.streaming:
        load_kwargs["streaming"] = True

    # Normalize standard language modeling / benchmark dataset identifiers
    dataset_id = name
    subset = config.subset

    if name_lower in ("wikitext", "wikitext2", "wikitext-2", "wikitext-2-raw-v1"):
        dataset_id = "Salesforce/wikitext"
        subset = subset or "wikitext-2-raw-v1"
    elif name_lower in ("wikitext103", "wikitext-103", "wikitext-103-raw-v1"):
        dataset_id = "Salesforce/wikitext"
        subset = subset or "wikitext-103-raw-v1"
    elif name_lower in ("ptb", "ptb_text_only"):
        dataset_id = "ptb_text_only"
    elif name_lower in ("c4", "allenai/c4"):
        dataset_id = "allenai/c4"
        subset = subset or "en"
        # C4 is massive; default to streaming unless user explicitly opted out
        if not load_kwargs.get("streaming", False) and config.streaming is not False:
            load_kwargs["streaming"] = True
    elif name_lower in ("openwebtext", "openwebtext-10k"):
        dataset_id = "openwebtext"

    try:
        if subset is not None:
            dataset = hf_load_dataset(dataset_id, subset, split=config.split, **load_kwargs)
        else:
            dataset = hf_load_dataset(dataset_id, split=config.split, **load_kwargs)
    except Exception as e:
        logger.warning("Primary load for '%s' (%s) failed: %s. Retrying without subset...", dataset_id, subset, e)
        try:
            dataset = hf_load_dataset(dataset_id, split=config.split, **load_kwargs)
        except Exception as fallback_e:
            raise RuntimeError(f"Could not load HuggingFace dataset '{name}': {fallback_e}") from fallback_e

    # If user requested streaming, return the iterable dataset directly
    if config.streaming:
        return dataset

    # For non-streaming DatasetDict, ensure validation split exists
    if isinstance(dataset, dict) and "validation" not in dataset and "test" not in dataset:
        if "train" in dataset and hasattr(dataset["train"], "train_test_split"):
            splits = dataset["train"].train_test_split(test_size=config.split_ratio)
            dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})
    elif hasattr(dataset, "train_test_split") and not isinstance(dataset, (dict, DatasetDict)):
        splits = dataset.train_test_split(test_size=config.split_ratio)
        dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    return dataset


def create_tokenized_dataloader(
    dataset: Any,
    tokenizer: Any,
    batch_size: int = 16,
    max_seq_length: int = 128,
    split: str = "train",
    task_type: str = "classification",
    is_lm: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_to_multiple_of: int = 8,
) -> DataLoader:
    """
    Build a DataLoader directly from any HuggingFace dataset with dynamic padding
    and Tensor Core alignment.

    Args:
        dataset: Dataset or DatasetDict.
        tokenizer: HuggingFace AutoTokenizer.
        batch_size: Micro-batch size.
        max_seq_length: Tokenization cutoff.
        split: Split key ('train', 'validation', 'test').
        task_type: 'classification', 'regression', or 'language_modeling'.
        is_lm: True if autoregressive/causal language modeling.
        shuffle: Whether to shuffle train samples.
        num_workers: DataLoader worker processes.
        pad_to_multiple_of: Multiple to pad sequence lengths (default 8).

    Returns:
        torch.utils.data.DataLoader
    """
    from neuralops.data.base import DataConfig, ProductionDataCollator, TaskType, preprocess_function

    active_ds = dataset[split] if isinstance(dataset, (dict, DatasetDict)) and split in dataset else dataset

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"

    t_type = TaskType.LANGUAGE_MODELING if (is_lm or task_type == "language_modeling") else TaskType.SINGLE_CLASSIFICATION
    cfg = DataConfig(
        task_type=t_type,
        max_length=max_seq_length,
        padding=False,  # dynamically collated
        truncation=True,
        batch_size=batch_size,
    )

    if hasattr(active_ds, "map"):
        def _tok_map(examples: dict[str, Any]) -> dict[str, Any]:
            return preprocess_function(examples, tokenizer, cfg)

        tokenized = active_ds.map(_tok_map, batched=True, remove_columns=active_ds.column_names)
    else:
        tokenized = active_ds

    collator = ProductionDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_seq_length,
    )

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(active_ds, IterableDataset) else False,
        num_workers=num_workers,
        collate_fn=collator,
    )


__all__ = [
    "ExperimentConfig",
    "load_dataset",
    "create_tokenized_dataloader",
]
