"""
Model‑agnostic data pipeline – dataset loading, tokenization, preprocessing,
DataLoader creation, and augmentation for all common tasks and models.
"""

from .augmentation import AUGMENTATION_REGISTRY, random_mask_augment
from .base import (
    DataConfig,
    ProductionDataCollator,
    TaskType,
    create_dataloader,
    preprocess_function,
)
from .benchmark_datasets import (
    BenchmarkDatasetInfo,
    BenchmarkDatasetRegistry,
    load_benchmark_dataset,
)
from .cached_tensors import get_cached_tokenized_dataset
from .dataset_loader import ExperimentConfig, load_dataset
from .glue_loader import (
    GLUE_TASK_KEYS,
    get_calibration_dataloader,
    get_num_labels_and_is_regression,
    load_glue_local,
)
from .multi_choice import (
    MultiChoiceDataset,
    convert_to_multichoice_format,
    is_multichoice_dataset,
)
from .synthetic_generator import generate_synthetic_data, run_universal_generation
from .task_mapper import get_metric_name, get_num_labels, get_text_columns
from .validation import DataValidator, ValidationReport

__all__ = [
    # Base
    "DataConfig",
    "ProductionDataCollator",
    "TaskType",
    "preprocess_function",
    "create_dataloader",
    "get_calibration_dataloader",
    # Benchmark datasets
    "BenchmarkDatasetInfo",
    "BenchmarkDatasetRegistry",
    "load_benchmark_dataset",
    # Dataset loading
    "ExperimentConfig",
    "load_dataset",
    # GLUE
    "GLUE_TASK_KEYS",
    "get_num_labels_and_is_regression",
    "load_glue_local",
    # Task helpers
    "get_metric_name",
    "get_num_labels",
    "get_text_columns",
    # Multi‑choice
    "MultiChoiceDataset",
    "is_multichoice_dataset",
    "convert_to_multichoice_format",
    # Synthetic
    "generate_synthetic_data",
    "run_universal_generation",
    # Validation
    "DataValidator",
    "ValidationReport",
    # Augmentation
    "AUGMENTATION_REGISTRY",
    "random_mask_augment",
]
