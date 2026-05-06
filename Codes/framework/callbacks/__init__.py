"""
Unified callback system for the Efficient Transformer Toolkit.
Every callback follows the HuggingFace TrainerCallback interface,
making them pluggable into any Trainer subclass.
"""

from typing import Optional, Dict, Any, List
import logging

# ----------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------
from .base import BaseCallback, CallbackList

# ----------------------------------------------------------------------
# Training callbacks
# ----------------------------------------------------------------------
from .training import (
    LoggingCallback,
    MetricsTrackerCallback,
    ProgressCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LRSchedulerCallback,
)

# ----------------------------------------------------------------------
# Resource monitoring
# ----------------------------------------------------------------------
from .resources import (
    MemoryCallback,
    TimeCallback,
    GPUStatsCallback,
)

# ----------------------------------------------------------------------
# Analysis & inspection
# ----------------------------------------------------------------------
from .analysis import (
    SVDAttentionCallback,
    AttentionHeatmapCallback,
    GradientNormCallback,
    LayerwiseAnalysisCallback,
    ActivationStatsCallback,
)

# ----------------------------------------------------------------------
# External monitoring integrations
# ----------------------------------------------------------------------
from .monitoring import (
    WandbCallback,
    TensorBoardCallback,
    TerminalCallback,
    WebSocketCallback,
)

# ----------------------------------------------------------------------
# Django integration (the bridge)
# ----------------------------------------------------------------------
from .django_integration import DjangoCallback

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    # Base
    "BaseCallback", "CallbackList",
    # Training
    "LoggingCallback", "MetricsTrackerCallback", "ProgressCallback",
    "EarlyStoppingCallback", "CheckpointCallback", "LRSchedulerCallback",
    # Resources
    "MemoryCallback", "TimeCallback", "GPUStatsCallback",
    # Analysis
    "SVDAttentionCallback", "AttentionHeatmapCallback", "GradientNormCallback",
    "LayerwiseAnalysisCallback", "ActivationStatsCallback",
    # Monitoring
    "WandbCallback", "TensorBoardCallback", "TerminalCallback", "WebSocketCallback",
    # Django
    "DjangoCallback",
]