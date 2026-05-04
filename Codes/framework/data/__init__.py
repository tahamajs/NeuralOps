# framework/data/__init__.py
from .dataset_loader import load_dataset, ExperimentConfig
from .glue_loader import load_glue_local, GLUE_TASK_KEYS, get_num_labels_and_is_regression
from .custom_loader import load_custom_dataset, load_dataset_from_file
from .multi_choice import MultiChoiceDataset, is_multichoice_dataset, convert_to_multichoice_format
from .preprocessing import preprocess_function
from .synthetic_generator import run_universal_generation, generate_synthetic_data
from .task_mapper import (
    TASK_METRICS, TASK_NUM_LABELS, TASK_COLUMNS,
    get_metric_name, get_num_labels, get_text_columns
)

__all__ = [
    "load_dataset",
    "ExperimentConfig",
    "load_glue_local",
    "GLUE_TASK_KEYS",
    "get_num_labels_and_is_regression",
    "load_custom_dataset",
    "load_dataset_from_file",
    "MultiChoiceDataset",
    "is_multichoice_dataset",
    "convert_to_multichoice_format",
    "preprocess_function",
    "run_universal_generation",
    "generate_synthetic_data",
    "TASK_METRICS",
    "TASK_NUM_LABELS",
    "TASK_COLUMNS",
    "get_metric_name",
    "get_num_labels",
    "get_text_columns",
]