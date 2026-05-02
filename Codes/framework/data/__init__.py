# framework/data/__init__.py
"""Data loading utilities for SVD-Lin framework."""
from .glue_loader import (
    load_glue_local,
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
)
