from .utils import (
    spectral_entropy,
    cumulative_entropy_fraction,
    allocate_ranks_spectron,
)
from .linear import LowRankSparseLinear
from .layer import (
    SpectronAttention,
    inject_spectron,
    SparsePruningCallback,
)

# Helpers/utilities for backward compatibility
from neuralops.layers.linear.spectron import (
    GLUE_TASK_KEYS,
    get_num_labels_and_type,
    preprocess,
)

__all__ = [
    "SpectronAttention",
    "LowRankSparseLinear",
    "SparsePruningCallback",
    "allocate_ranks_spectron",
    "inject_spectron",
    "spectral_entropy",
    "cumulative_entropy_fraction",
    "GLUE_TASK_KEYS",
    "get_num_labels_and_type",
    "preprocess",
]
