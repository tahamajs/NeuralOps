"""
Model‑agnostic data pipeline – dataset loading, tokenization, preprocessing,
DataLoader creation, and augmentation for all common tasks and models.
"""

from .base import (
    DataConfig, TaskType, preprocess_function as base_preprocess, create_dataloader
)
from .dataset_loader import ExperimentConfig, load_dataset
from .glue_loader import (
    GLUE_TASK_KEYS, get_num_labels_and_is_regression, load_glue_local
)
from .task_mapper import get_metric_name, get_num_labels, get_text_columns
from .multi_choice import MultiChoiceDataset, is_multichoice_dataset, convert_to_multichoice_format
from .synthetic_generator import generate_synthetic_data, run_universal_generation
from .validation import DataValidator, ValidationReport
from .augmentation import AUGMENTATION_REGISTRY, random_mask_augment

__all__ = [
    # Base
    "DataConfig", "TaskType", "preprocess_function", "create_dataloader",
    # Dataset loading
    "ExperimentConfig", "load_dataset",
    # GLUE
    "GLUE_TASK_KEYS", "get_num_labels_and_is_regression", "load_glue_local",
    # Task helpers
    "get_metric_name", "get_num_labels", "get_text_columns",
    # Multi‑choice
    "MultiChoiceDataset", "is_multichoice_dataset", "convert_to_multichoice_format",
    # Synthetic
    "generate_synthetic_data", "run_universal_generation",
    # Validation
    "DataValidator", "ValidationReport",
    # Augmentation
    "AUGMENTATION_REGISTRY", "random_mask_augment",
]