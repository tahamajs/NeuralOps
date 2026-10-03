Module neuralops.data.dataset_loader
====================================
Unified dataset loader driven by an ExperimentConfig.
Supports GLUE, custom CSV/JSON/Parquet, multi‑choice, synthetic, and HuggingFace hub datasets.

Functions
---------

`load_dataset(config: neuralops.data.dataset_loader.ExperimentConfig) ‑> datasets.dataset_dict.DatasetDict`
:   Load a dataset based on ExperimentConfig.
    Returns DatasetDict with at least 'train' and optionally 'validation'/'test' splits.

Classes
-------

`ExperimentConfig(dataset_name: str, dataset_path: str | None = None, task_type: str = 'single', text_columns: list[str] | None = None, label_column: str | None = None, num_labels: int = 2, metric: str = 'accuracy', max_seq_length: int = 128, split_ratio: float = 0.1, synthetic_samples: int = 5000, glue_base_path: str | None = None, **kwargs)`
:   Configuration for loading a dataset.
    
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