"""Backward-compatibility shim. Moved to neuralops.core.utils.gqa."""
import warnings
warnings.warn(
    "neuralops.core.gqa_utils is deprecated. Use neuralops.core.utils.gqa.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.core.utils.gqa import *  # noqa: F401,F403
from neuralops.core.utils.gqa import repeat_kv  # noqa: F401

repeat_kv_optimized = repeat_kv
