"""Backward-compatibility shim.

This module has moved to neuralops.core.compression_pipeline.
Update your imports:

    from neuralops.core.compression_pipeline import CompressionPipeline
"""
import warnings

warnings.warn(
    "neuralops.core.compression_engine has moved to "
    "neuralops.core.compression_pipeline. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.core.compression_pipeline import *  # noqa: F401,F403
from neuralops.core.compression_pipeline import (  # noqa: F401
    CompressionPipeline,
    compress_model,
)
