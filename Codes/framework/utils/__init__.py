# framework/utils/__init__.py
"""
Utility functions for SVD-Lin framework.

This module provides:
- Device management (auto-detection, memory clearing)
- Logging setup
- Model profiling (memory, time, FLOPs)
- Metrics computation (accuracy, correlation, etc.)
- Checkpoint management
- Architecture detection (BERT, RoBERTa, DistilBERT, etc.)
- Configuration loading
- General helper functions

Example:
    >>> from framework.utils import get_device, setup_logger, ModelProfiler
    >>> 
    >>> device = get_device()
    >>> logger = setup_logger("my_experiment")
    >>> profiler = ModelProfiler(model)
    >>> profiler.start_profiling()
    >>> # ... run model ...
    >>> report = profiler.get_report()
"""

from .device import get_device, clear_memory, get_memory_usage, get_model_device
from .logger import setup_logger, get_logger
from .profiler import ModelProfiler, measure_inference_latency, measure_memory_usage
from .metrics import (
    compute_accuracy,
    compute_matthews_corrcoef,
    compute_pearson,
    compute_spearman,
    compute_f1,
    compute_precision_recall,
    compute_metrics_for_task,
)
from .checkpoint import CheckpointManager
from .architecture import (
    ARCH_PATTERNS,
    detect_architecture,
    get_layer_patterns,
    get_compression_targets,
    get_model_info,
)
from .config import load_config, save_config, update_config
from .helpers import (
    set_seed,
    format_time,
    format_number,
    count_parameters,
    ensure_dir,
    save_json,
    load_json,
    flatten_dict,
    unflatten_dict,
)

__all__ = [
    # Device
    "get_device",
    "clear_memory",
    "get_memory_usage",
    "get_model_device",
    
    # Logging
    "setup_logger",
    "get_logger",
    
    # Profiling
    "ModelProfiler",
    "measure_inference_latency",
    "measure_memory_usage",
    
    # Metrics
    "compute_accuracy",
    "compute_matthews_corrcoef",
    "compute_pearson",
    "compute_spearman",
    "compute_f1",
    "compute_precision_recall",
    "compute_metrics_for_task",
    
    # Checkpoint
    "CheckpointManager",
    
    # Architecture
    "ARCH_PATTERNS",
    "detect_architecture",
    "get_layer_patterns",
    "get_compression_targets",
    "get_model_info",
    
    # Config
    "load_config",
    "save_config",
    "update_config",
    
    # Helpers
    "set_seed",
    "format_time",
    "format_number",
    "count_parameters",
    "ensure_dir",
    "save_json",
    "load_json",
    "flatten_dict",
    "unflatten_dict",
]

__version__ = "1.0.0"