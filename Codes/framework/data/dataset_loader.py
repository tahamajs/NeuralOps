"""
Unified dataset loader driven by an ExperimentConfig.
Supports GLUE, custom CSV/JSON/Parquet, multi‑choice, synthetic, and more.
"""

import os
from pathlib import Path
from typing import Optional, List
from datasets import DatasetDict


class ExperimentConfig:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str] = None,
        task_type: str = "single",          # "single", "pair", "multi_choice", "regression"
        text_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
        num_labels: int = 2,
        metric: str = "accuracy",
        max_seq_length: int = 128,
        split_ratio: float = 0.1,           # for custom datasets without validation split
        synthetic_samples: int = 5000,
        glue_base_path: Optional[str] = None,
        **kwargs
    ):
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
        self.glue_base_path = glue_base_path or "./data/glue"
        self.kwargs = kwargs


def load_dataset(config: ExperimentConfig) -> DatasetDict:
    """
    Load a dataset based on ExperimentConfig.

    Returns:
        DatasetDict with at least 'train' and optionally 'validation'/'test' splits.
    """
    name = config.dataset_name.lower()

    if name in {"cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"}:
        return _load_glue(config)
    elif name == "synthetic":
        return _load_synthetic(config)
    elif name.endswith((".csv", ".json", ".jsonl", ".parquet")):
        return _load_custom(config)
    elif name == "multi_choice":
        return _load_multi_choice(config)
    else:
        # fallback: HuggingFace hub
        from datasets import load_dataset as hf_load
        dataset = hf_load(name)
        if "validation" not in dataset and "test" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=config.split_ratio)
        return dataset


def _load_glue(config):
    from .glue_loader import load_glue_local
    return load_glue_local(config.dataset_name, config.glue_base_path)


def _load_synthetic(config):
    import tempfile
    from .synthetic_generator import run_universal_generation
    from datasets import Dataset
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_universal_generation(
            output_dir=Path(tmpdir),
            total_samples_per_domain=config.synthetic_samples
        )
        dataset = Dataset.from_pandas(df)
        splits = dataset.train_test_split(test_size=config.split_ratio)
        return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def _load_custom(config):
    from .custom_loader import load_dataset_from_file
    return load_dataset_from_file(
        file_path=config.dataset_path or config.dataset_name,
        split_ratio=config.split_ratio,
        text_column=config.text_columns[0] if config.text_columns else None,
        label_column=config.label_column
    )


def _load_multi_choice(config):
    from .custom_loader import load_custom_dataset
    return load_custom_dataset(config.dataset_path)