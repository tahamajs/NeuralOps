# framework/data/__init__.py
"""
Data loading and preprocessing utilities for SVD-Lin framework.

This module provides:
- GLUE dataset loading from local Parquet files
- Custom dataset loading (CSV, JSON, Parquet)
- Multi-choice dataset handling
- Synthetic data generation
- Task-specific preprocessing
- Dataset discovery and configuration

Example:
    >>> from framework.data import load_glue_task, load_custom_dataset
    >>> 
    >>> # Load GLUE task
    >>> dataset = load_glue_task("cola", "/path/to/glue")
    >>> 
    >>> # Load custom dataset
    >>> dataset = load_custom_dataset("data.csv", tokenizer)
"""

from .glue_loader import load_glue_task, load_glue_tasks, get_glue_task_info
from .custom_loader import load_custom_dataset, load_dataset_from_file
from .multi_choice import MultiChoiceDataset, is_multichoice_dataset
from .synthetic_generator import run_universal_generation, generate_synthetic_data
from .preprocessing import preprocess_function, tokenize_dataset, DataCollator
from .task_config import (
    TASK_METRICS,
    TASK_NUM_LABELS,
    TASK_COLUMNS,
    get_metric_name,
    get_num_labels,
    get_text_columns,
)
from .utils import (
    discover_datasets,
    split_dataset,
    balance_dataset,
    filter_by_length,
    DatasetInfo
)

__all__ = [
    # GLUE loading
    "load_glue_task",
    "load_glue_tasks",
    "get_glue_task_info",
    
    # Custom loading
    "load_custom_dataset",
    "load_dataset_from_file",
    
    # Multi-choice
    "MultiChoiceDataset",
    "is_multichoice_dataset",
    
    # Synthetic
    "run_universal_generation",
    "generate_synthetic_data",
    
    # Preprocessing
    "preprocess_function",
    "tokenize_dataset",
    "DataCollator",
    
    # Task config
    "TASK_METRICS",
    "TASK_NUM_LABELS",
    "TASK_COLUMNS",
    "get_metric_name",
    "get_num_labels",
    "get_text_columns",
    
    # Utilities
    "discover_datasets",
    "split_dataset",
    "balance_dataset",
    "filter_by_length",
    "DatasetInfo",
]

__version__ = "1.0.0"