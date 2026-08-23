"""GLUE benchmark task definitions, input field mappings, and label metadata."""

from __future__ import annotations
from typing import Dict, Any, Tuple

GLUE_TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sst2": {
        "hf_name": "sst2",
        "keys": ("sentence", None),
        "num_labels": 2,
        "metric_key": "accuracy",
        "is_regression": False,
    },
    "mrpc": {
        "hf_name": "mrpc",
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric_key": "f1",
        "is_regression": False,
    },
    "qnli": {
        "hf_name": "qnli",
        "keys": ("question", "sentence"),
        "num_labels": 2,
        "metric_key": "accuracy",
        "is_regression": False,
    },
    "rte": {
        "hf_name": "rte",
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric_key": "accuracy",
        "is_regression": False,
    },
    "cola": {
        "hf_name": "cola",
        "keys": ("sentence", None),
        "num_labels": 2,
        "metric_key": "mcc",
        "is_regression": False,
    },
    "stsb": {
        "hf_name": "stsb",
        "keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "metric_key": "pearson",
        "is_regression": True,
    },
    "qqp": {
        "hf_name": "qqp",
        "keys": ("question1", "question2"),
        "num_labels": 2,
        "metric_key": "f1",
        "is_regression": False,
    },
    "mnli": {
        "hf_name": "mnli",
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "metric_key": "accuracy",
        "is_regression": False,
    },
}


def get_task_info(task_name: str) -> Dict[str, Any]:
    """Retrieve configuration metadata for a GLUE task identifier."""
    clean_name = task_name.lower().replace("-", "")
    if clean_name in GLUE_TASK_CONFIGS:
        return GLUE_TASK_CONFIGS[clean_name]
    # Default fallback to SST-2 schema
    return GLUE_TASK_CONFIGS["sst2"]

from .dataset_loader import load_and_tokenize_glue_task  # re-export for backwards compatibility
