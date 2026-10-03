"""
Unified dataset loader driven by an ExperimentConfig.
Supports GLUE, custom CSV/JSON/Parquet, multi‑choice, synthetic, and HuggingFace hub datasets.
"""

from pathlib import Path

try:
    from datasets import DatasetDict
except (ImportError, AttributeError):
    DatasetDict = dict


class ExperimentConfig:
    """
    Configuration for loading a dataset.

    Attributes:
        dataset_name: Name of the dataset (GLUE task, synthetic, file path, or HF identifier).
        dataset_path: Path to local dataset file (for custom or multi‑choice).
        task_type: "single", "pair", "multi_choice", "regression".
        text_columns: List of column names containing text (auto‑detected if None).
        label_column: Name of the label column (auto‑detected if None).
        num_labels: Number of classes (for classification).
        metric: Primary metric for evaluation (e.g., "accuracy").
        max_seq_length: Maximum sequence length for tokenization.
        split_ratio: Fraction of data to use for validation (for custom datasets).
        synthetic_samples: Number of synthetic samples to generate (if dataset_name="synthetic").
        glue_base_path: Base directory for local GLUE Parquet files.
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
        **kwargs
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
        from framework.utils.paths import get_glue_dir

        self.glue_base_path = glue_base_path or str(get_glue_dir())
        self.kwargs = kwargs


def load_dataset(config: ExperimentConfig) -> DatasetDict:
    """
    Load a dataset based on ExperimentConfig.
    Returns DatasetDict with at least 'train' and optionally 'validation'/'test' splits.
    """
    name = config.dataset_name.lower()

    # GLUE tasks (local Parquet)
    if name in {"cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"}:
        return _load_glue(config)

    # Synthetic data generation
    elif name == "synthetic":
        return _load_synthetic(config)

    # Custom file (CSV, JSON, Parquet)
    elif name.endswith((".csv", ".json", ".jsonl", ".parquet")):
        return _load_custom(config)

    # Multi‑choice dataset (ARC, HellaSwag, etc.)
    elif name == "multi_choice":
        return _load_multi_choice(config)

    # Fallback: HuggingFace dataset hub
    else:
        return _load_hub(config)


def _load_glue(config: ExperimentConfig) -> DatasetDict:
    """ load glue."""
    from .glue_loader import load_glue_local

    return load_glue_local(config.dataset_name, config.glue_base_path)


def _load_synthetic(config: ExperimentConfig) -> DatasetDict:
    """ load synthetic."""
    import tempfile

    from datasets import Dataset

    from .synthetic_generator import run_universal_generation

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_universal_generation(
            output_dir=Path(tmpdir), total_samples_per_domain=config.synthetic_samples
        )
        dataset = Dataset.from_pandas(df)
        splits = dataset.train_test_split(test_size=config.split_ratio)
        return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def _load_custom(config: ExperimentConfig) -> DatasetDict:
    """ load custom."""
    from .custom_loader import load_dataset_from_file

    file_path = config.dataset_path or config.dataset_name
    text_col = config.text_columns[0] if config.text_columns else None
    return load_dataset_from_file(
        file_path=file_path,
        split_ratio=config.split_ratio,
        text_column=text_col,
        label_column=config.label_column,
        **config.kwargs
    )


def _load_multi_choice(config: ExperimentConfig) -> DatasetDict:
    """ load multi choice."""
    from .custom_loader import load_custom_dataset

    # Multi‑choice datasets are loaded as raw; conversion to MultiChoiceDataset is done later.
    return load_custom_dataset(config.dataset_path or config.dataset_name)


def _load_hub(config: ExperimentConfig) -> DatasetDict:
    """ load hub."""
    from datasets import load_dataset as hf_load

    dataset = hf_load(config.dataset_name, **config.kwargs)
    if "validation" not in dataset and "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=config.split_ratio)
    return dataset
